from pointnet._dir_setting_ import *
import pandas as pd
pd.set_option('mode.chained_assignment', None) # disable SettingWithCopyWarning


import sys
import os

import cv2
import numpy as np

from natsort import natsorted
import glob

import pydicom
#from configure import *

#--helper ---

class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


############################################################################
# data processing

series_description_map = {
    'Sagittal T2/STIR': 'sagittal_t2',
    'Sagittal T1': 'sagittal_t1',
    'Axial T2':'axial_t2',
}
grade_map = {
    'Missing': -1,
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2,
}
condition_map = { #follow sample submission order
    'Left Neural Foraminal Narrowing':  'left_neural_foraminal_narrowing',
    'Left Subarticular Stenosis':       'left_subarticular_stenosis',
    'Right Neural Foraminal Narrowing': 'right_neural_foraminal_narrowing',
    'Right Subarticular Stenosis':      'right_subarticular_stenosis',
    'Spinal Canal Stenosis':            'spinal_canal_stenosis',
}
level_map = {
    'L1/L2': 'l1_l2',
    'L2/L3': 'l2_l3',
    'L3/L4': 'l3_l4',
    'L4/L5': 'l4_l5',
    'L5/S1': 'l5_s1',
}

condition_level_col=[ #follow sample submission order
    'left_neural_foraminal_narrowing_l1_l2',
    'left_neural_foraminal_narrowing_l2_l3',
    'left_neural_foraminal_narrowing_l3_l4',
    'left_neural_foraminal_narrowing_l4_l5',
    'left_neural_foraminal_narrowing_l5_s1',
    'left_subarticular_stenosis_l1_l2',
    'left_subarticular_stenosis_l2_l3',
    'left_subarticular_stenosis_l3_l4',
    'left_subarticular_stenosis_l4_l5',
    'left_subarticular_stenosis_l5_s1',
    'right_neural_foraminal_narrowing_l1_l2',
    'right_neural_foraminal_narrowing_l2_l3',
    'right_neural_foraminal_narrowing_l3_l4',
    'right_neural_foraminal_narrowing_l4_l5',
    'right_neural_foraminal_narrowing_l5_s1',
    'right_subarticular_stenosis_l1_l2',
    'right_subarticular_stenosis_l2_l3',
    'right_subarticular_stenosis_l3_l4',
    'right_subarticular_stenosis_l4_l5',
    'right_subarticular_stenosis_l5_s1',
    'spinal_canal_stenosis_l1_l2',
    'spinal_canal_stenosis_l2_l3',
    'spinal_canal_stenosis_l3_l4',
    'spinal_canal_stenosis_l4_l5',
    'spinal_canal_stenosis_l5_s1',
]
condition_level_classname=['none']+condition_level_col

########################################################################################

def np_dot(a,b):
    return  (a*b).sum(-1)

# project 2d to 3d
def project_to_3d(x,y,z, df):
    d = df.iloc[z]
    H, W = d.H, d.W
    sx, sy, sz = [float(v) for v in d.ImagePositionPatient]
    o0, o1, o2, o3, o4, o5, = [float(v) for v in d.ImageOrientationPatient]
    delx, dely = d.PixelSpacing

    xx = o0 * delx * x + o3 * dely * y + sx
    yy = o1 * delx * x + o4 * dely * y + sy
    zz = o2 * delx * x + o5 * dely * y + sz
    return xx,yy,zz

# back project 3d to 2d
def backproject_to_2d(xx,yy,zz,df):

    d = df.iloc[0]
    sx, sy, sz = [float(v) for v in d.ImagePositionPatient]
    o0, o1, o2, o3, o4, o5, = [float(v) for v in d.ImageOrientationPatient]
    delx, dely = d.PixelSpacing
    delz = d.SpacingBetweenSlices

    ox = np.array([o0,o1,o2])
    oy = np.array([o3,o4,o5])
    oz = np.cross(ox,oy)

    p = np.array([xx-sx,yy-sy,zz-sz])
    x = np.dot(ox, p)/delx
    y = np.dot(oy, p)/dely
    z = np.dot(oz, p)/delz
    x = int(round(x))
    y = int(round(y))
    z = int(round(z))


    D,H,W = len(df),d.H,d.W
    inside = \
        (x>=0) & (x<W) &\
        (y>=0) & (y<H) &\
        (z>=0) & (z<D)
    if not inside:
        #print('out-of-bound')
        return False,0,0,0,0

    n = df.instance_number.values[int(round(z))]
    return True,x,y,z,n


def draw_slice(ax, df, color=[[1,0,0]], alpha=[0.1]):
    if len(color)==1:
        color = color*len(df)
    if len(alpha)==1:
        alpha = alpha*len(df)

    num_slice = len(df)
    for i in range(num_slice):
        d = df.iloc[i]
        W, H = d.W, d.H
        o0, o1, o2, o3, o4, o5 = d.ImageOrientationPatient
        ox = np.array([o0, o1, o2])
        oy = np.array([o3, o4, o5])
        sx, sy, sz = d.ImagePositionPatient
        s = np.array([sx, sy, sz])
        delx, dely = d.PixelSpacing

        p0 = s
        p1 = s + W * delx * ox
        p2 = s + H * dely * oy
        p3 = s + H * dely * oy + W * delx * ox

        grid = np.stack([p0, p1, p2, p3]).reshape(2, 2, 3)
        gx = grid[:, :, 0]
        gy = grid[:, :, 1]
        gz = grid[:, :, 2]

        ax.plot_surface(gx, gy, gz, alpha=alpha[i], color=color[i])
        if i==0:
            ax.scatter([sx], [sy], [sz], color='black')
        else:
            ax.scatter([sx], [sy], [sz],  alpha=alpha[i], color='black')


########################################################################################

def normalise_to_8bit(x, lower=0.1, upper=99.9): # 1, 99 #0.05, 99.5 #0, 100
    lower, upper = np.percentile(x, (lower, upper))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)

def load_mri_from_dicom_dir(
    study_id,
    series_id,
    series_description,
    is_split=False,
):
    data_kaggle_dir = DATA_KAGGLE_DIR
    dicom_dir= f'{data_kaggle_dir}/train_images/{study_id}/{series_id}'

    #---
    dicom_file = natsorted(glob.glob( f'{dicom_dir}/*.dcm'))
    instance_number = [int(f.split('/')[-1].split('.')[0]) for f in dicom_file]
    dicom = [pydicom.dcmread(f) for f in dicom_file]
    H,W = dicom[0].pixel_array.shape

    dicom_df = []
    for i,d in zip(instance_number,dicom): #d__.dict__
        dicom_df.append(
            dotdict(
                study_id=study_id,
                series_id=series_id,
                series_description=series_description,
                instance_number=i,
                #InstanceNumber = d.InstanceNumber,
                ImagePositionPatient=[float(v) for v in d.ImagePositionPatient],
                ImageOrientationPatient=[float(v) for v in d.ImageOrientationPatient],
                PixelSpacing=[float(v) for v in d.PixelSpacing],
                SpacingBetweenSlices=float(d.SpacingBetweenSlices),
                SliceThickness=float(d.SliceThickness),
                grouping=str([round(float(v),3) for v in d.ImageOrientationPatient]),
                H=H,
                W=W,
            )
        )
    dicom_df = pd.DataFrame(dicom_df)
    dicom_df = [d for _,d in dicom_df.groupby('grouping')]
    #print('ok')

    #sort slice
    mri=[]
    mri_group_sort=[]
    for df in dicom_df:
        position = np.array(df['ImagePositionPatient'].values.tolist())
        orientation = np.array(df['ImageOrientationPatient'].values.tolist())
        normal = np.cross(orientation[:,:3], orientation[:,3:])
        projection = np.sum(normal*position,1) #np.dot(normal, position)
        df.loc[:,'projection'] = projection
        df = df.sort_values('projection')
        df.loc[:,'z'] = df.index
        #print('ok')

        # todo: assert all slices are continous
        # use  (position[-1]-position[0])/N = SpacingBetweenSlices ??
        assert len(df.SliceThickness.unique())==1
        assert len(df.SpacingBetweenSlices.unique())==1
        #assert len(df.ImageOrientationPatient.unique())==1

        volume = [
            dicom[instance_number.index(i)].pixel_array for i in df.instance_number
        ]
        volume = np.stack(volume)
        volume = normalise_to_8bit(volume)
        mri.append(dotdict(
            df=df,
            volume=volume,
        ))

        if 'sagittal'in series_description.lower():
            mri_group_sort.append(position[0,0]) #x
        if 'axial' in series_description.lower():
            mri_group_sort.append(position[0,2]) #z

    mri = [r for _, r in sorted(zip(mri_group_sort, mri))]
    for i,r in enumerate(mri):
        r.df.loc[:,'group']=i
    if is_split ==False:
        mri = dotdict(
            series_id = series_id,
            df = pd.concat([r.df for r in mri]),
            volume = np.concatenate([r.volume for r in mri]),
        )
    return mri

# add shape W,H; world coords xx,yy,zz
def add_XYZ_to_label_df(label_df):

    for col in ['W','H']:
        label_df.loc[:,col]=0
    for col in ['xx','yy','zz']:
        label_df.loc[:,col]=0.0

    for t,d in label_df.iterrows():
        dicom_file = f'{DATA_KAGGLE_DIR}/train_images/{d.study_id}/{d.series_id}/{d.instance_number}.dcm'
        dicom = pydicom.dcmread(dicom_file)
        H,W = dicom.pixel_array.shape
        sx,sy,sz = [float(v) for v in dicom.ImagePositionPatient]
        o0, o1, o2, o3, o4, o5, = [float(v) for v in dicom.ImageOrientationPatient]
        delx,dely = dicom.PixelSpacing

        xx =  o0*delx*d.x + o3*dely*d.y + sx
        yy =  o1*delx*d.x + o4*dely*d.y + sy
        zz =  o2*delx*d.x + o5*dely*d.y + sz

        label_df.loc[t,'W'] = W
        label_df.loc[t,'H'] = H
        label_df.loc[t,'xx'] = xx
        label_df.loc[t,'yy'] = yy
        label_df.loc[t,'zz'] = zz

    return label_df


##################################################################################################
def load_dummy_df():
    study_id = [8785691]
    valid_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_series_descriptions.csv')
    valid_df = valid_df[
        (valid_df.study_id.isin (study_id))
        #&(valid_df.series_description == 'Axial T2')
    ].reset_index(drop=True)
    valid_df.loc[:,'series_description'] = valid_df.series_description.map(series_description_map)
    return valid_df

def load_dummy_data(valid_df):
    valid_data = []
    for study_id in valid_df['study_id'].unique():

        data = dotdict(
            study_id = study_id,
            axial_t2 = [],
            sagittal_t2 = [],
            sagittal_t1 = [],
        )
        for i,d in valid_df.iterrows():
            series_id = d.series_id
            series_description = d.series_description

            # here we assume there is one mri for each view
            # todo: fix for multiple mri
            data[series_description].append(
                load_mri_from_dicom_dir(
                    study_id,
                    series_id,
                    series_description,
                    is_split=False,
                )
            )
        valid_data.append(data)

    return valid_data

def data_to_text(data):
    text=''
    text+=f'study_id: {data.study_id}\n'

    for series_description in ['axial_t2','sagittal_t2', 'sagittal_t1']:
        text+=f'[{series_description}]: {len(data[series_description])}\n'

        for r in data[series_description]:
            text+=f'\tseries_id: {r.series_id}\n'
            df = r.df
            volume = r.volume
            num_group =df['group'].nunique()
            text += f'\t\tnum_group: {num_group}\n'
            text += f'\t\tvolume:{volume.shape}\n'
    text +='\n'
    return text


#---
# def fill_sphere(volume, center, radius, fill):
#     cx,cy,cz = center
#     for x in range(cx-radius, cx+radius+1):
#         for y in range(cy-radius, cy+radius+1):
#             for z in range(cz-radius, cz+radius+1):
#                 #d = radius - abs(cx-x) - abs(cy-y) - abs(cz-z)
#                 d = radius - ((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2) ** 0.5
#                 if d>=0: volume[x,y,z] = fill
#     return volume

def make_mask(mask, center, radius_xy, fill):
    cx,cy,cz = center
    cv2.circle(mask[cz],(cx,cy),radius_xy,fill,-1,cv2.LINE_4)
    return mask

def load_truth(valid_data):
    study_id = [ d.study_id for d  in valid_data ]

    grade_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train.csv')
    grade_df = grade_df.set_index('study_id')
    grade_df = grade_df.fillna('Missing')
    grade_df = grade_df[condition_level_col]
    grade_df = grade_df.map(lambda x:grade_map[x])

    label_coord_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_label_coordinates.csv')

    valid_truth = []
    for d in valid_data:
        study_id = d.study_id
        grade = grade_df.loc[study_id].values

        point_df = label_coord_df[label_coord_df.study_id==study_id]
        point_df = add_XYZ_to_label_df(point_df)
        point_df = point_df.sort_values(['condition', 'level'])
        assert(len(point_df)==25)
        #print(point_df)

        #---
        #todo: support multi series??
        r = d.sagittal_t2[0]
        series_id = r.series_id
        D,H,W = r.volume.shape
        mask = np.zeros((D,H,W),dtype=np.uint8)

        point = []
        for i,d in point_df.iterrows():
            #print(d.xx,d.yy,d.zz)

            inside, x,y,z,n = backproject_to_2d(d.xx, d.yy, d.zz, r.df)
            assert(inside)

            classlabel = condition_level_classname.index(condition_map[d.condition] + '_' + level_map[d.level])
            #print(classlabel)# running order from 1 to 25

            center = x,y,z
            mask = make_mask(mask, center, radius_xy=5, fill=classlabel)
            point.append(( x,y,z))

        point = np.stack(point)
        truth = dotdict(
            study_id=study_id,
            grade=grade,
            point_df=point_df,
            sagittal_t2=[dotdict(
                series_id=series_id,
                mask=mask,
                point=point, #
            )],
        )
        valid_truth.append(truth)
    return valid_truth

def truth_to_text(truth):
    text=''
    text+=f'study_id: {truth.study_id}\n'
    text+=f'grade: {truth.grade.shape}\n'
    text+=f'point_df: {len(truth.point_df)}\n'

    for series_description in ['sagittal_t2']:
        text+=f'[{series_description}]: {len(truth[series_description])}\n'

        for t in truth[series_description]:
            text+=f'\tseries_id: {t.series_id}\n'
            text += f'\t\tmask:{t.mask.shape}\n'
            text += f'\t\tpoint:{t.point.shape}\n'
    text +='\n'
    return text

