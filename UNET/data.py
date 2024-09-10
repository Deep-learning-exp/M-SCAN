from pointnet._dir_setting_ import *
from natsort import natsorted
import pandas as pd
pd.set_option('mode.chained_assignment', None) # disable SettingWithCopyWarning

import numpy as np
import pydicom
import glob

import cv2

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

################################################################################3


# read into volume (np_array) + dicom_header(df)
'''
import notes:
- instance_number may not be sequentially (can have missing num)
- instance_number is 1-indexed

'''

## 3d/2d processing #########################################################
def np_dot(a,b):
	return np.sum(a * b, 1)

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

## read data #########################################################
def resize_volume(volume, image_size):
	image = volume.copy()
	image = np.ascontiguousarray(image.transpose((1, 2, 0)))
	image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
	image = np.ascontiguousarray(image.transpose((2, 0, 1)))  # cv2.INTER_LINEAR=1
	return image

def normalise_to_8bit(x, lower=0.1, upper=99.9): # 1, 99 #0.05, 99.5 #0, 100
	lower, upper = np.percentile(x, (lower, upper))
	x = np.clip(x, lower, upper)
	x = x - np.min(x)
	x = x / np.max(x)
	return (x * 255).astype(np.uint8)
def read_series(study_id,series_id,series_description):
	data_kaggle_dir = DATA_KAGGLE_DIR
	dicom_dir = f'{data_kaggle_dir}/train_images/{study_id}/{series_id}'

	# read dicom file
	dicom_file = natsorted(glob.glob(f'{dicom_dir}/*.dcm'))
	#instance_number = [int(f.split('/')[-1].split('.')[0]) for f in dicom_file]
	#For windowa - 
	instance_number = [int(f.split('/')[-1].split('\\')[0].split('.')[0]) for f in dicom_file]
	dicom = [pydicom.dcmread(f) for f in dicom_file]

	# make dicom header df
	H, W = dicom[0].pixel_array.shape
	dicom_df = []
	for i, d in zip(instance_number, dicom):  # d__.dict__
		dicom_df.append(
			dotdict(
				study_id=study_id,
				series_id=series_id,
				series_description=series_description,
				instance_number=i,
				# InstanceNumber = d.InstanceNumber,
				ImagePositionPatient=[float(v) for v in d.ImagePositionPatient],
				ImageOrientationPatient=[float(v) for v in d.ImageOrientationPatient],
				PixelSpacing=[float(v) for v in d.PixelSpacing],
				SpacingBetweenSlices=float(d.SpacingBetweenSlices),
				SliceThickness=float(d.SliceThickness),
				grouping=str([round(float(v), 3) for v in d.ImageOrientationPatient]),
				H=H,
				W=W,
			)
		)
	dicom_df = pd.DataFrame(dicom_df)

	# sort slices
	dicom_df = [d for _, d in dicom_df.groupby('grouping')]

	data = []
	sort_data_by_group = []
	for df in dicom_df:
		position = np.array(df['ImagePositionPatient'].values.tolist())
		orientation = np.array(df['ImageOrientationPatient'].values.tolist())
		normal = np.cross(orientation[:, :3], orientation[:, 3:])
		projection = np_dot(normal, position)
		df.loc[:, 'projection'] = projection
		df = df.sort_values('projection')


		# todo: assert all slices are continous ??
		# use  (position[-1]-position[0])/N = SpacingBetweenSlices ??
		assert len(df.SliceThickness.unique()) == 1
		assert len(df.SpacingBetweenSlices.unique()) == 1


		volume = [
			dicom[instance_number.index(i)].pixel_array for i in df.instance_number
		]
		volume = np.stack(volume)
		volume = normalise_to_8bit(volume)
		data.append(dotdict(
			df=df,
			volume=volume,
		))

		if 'sagittal' in series_description.lower():
			sort_data_by_group.append(position[0, 0])  # x
		if 'axial' in series_description.lower():
			sort_data_by_group.append(position[0, 2])  # z

	data = [r for _, r in sorted(zip(sort_data_by_group, data))]
	for i, r in enumerate(data):
		r.df.loc[:, 'group'] = i

	df = pd.concat([r.df for r in data])
	df.loc[:, 'z'] = np.arange(len(df))
	try:
		volume = np.concatenate([r.volume for r in data])
	except:
		#remove those with different size
		volume = np.concatenate([r.volume for r in data if r.volume.shape[1]==data[0].volume.shape[1]])
	data = dotdict(
		series_id=series_id,
		df=df,
		volume=volume,
	)
	return data
def read_study(study_id, sagittal_t2_id, axial_t2_id):
	return dotdict(
		study_id = study_id,
		sagittal_t2 =read_series(study_id, sagittal_t2_id, 'sagittal_t2'),
		axial_t2 =read_series(study_id, axial_t2_id, 'axial_t2'),
	)
def get_true_sagittal_t2_point(study_id, sagittal_t2_df):
	series_id = sagittal_t2_df.iloc[0].series_id
	label_coord_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_label_coordinates.csv')
	label_df = label_coord_df[
		 (label_coord_df.study_id == study_id)
	   & (label_coord_df.series_id == series_id)
	]
	label_df = label_df.sort_values('level')
	point=label_df[['x','y']].values
	instance_number = label_df.instance_number.values

	#mapping from instance num to z (array index)
	map_instance_number, map_z = sagittal_t2_df[['instance_number','z',]].values.T
	map = {n:z for n,z in zip(map_instance_number,map_z)}
	z = [map[n] for n in instance_number]
	return point,z

############################################################3
#post processing (sagittal_t2 point net)
def probability_to_point(probability, threshold=0.0):
	#todo: handle mssing point
	#Missing point 
	
	point=[]
	for l in range(1, 6):
		y, x = np.where(probability[l] > threshold)
		y = round(y.mean())
		x = round(x.mean())
		point.append((x, y))
	return point


def view_to_world(sagittal_t2_point, z, sagittal_t2_df, image_size):

	H = sagittal_t2_df.iloc[0].H
	W = sagittal_t2_df.iloc[0].W
	scale_x = W / image_size
	scale_y = H / image_size

	xxyyzz = []
	for l in range(1, 6):
		x,y = sagittal_t2_point[l-1]
		xx,yy,zz = project_to_3d(x*scale_x, y*scale_y, z, sagittal_t2_df)
		xxyyzz.append((xx, yy, zz))

	xxyyzz = np.array(xxyyzz)
	return xxyyzz

def point_to_level(world_point, axial_t2_df):

	# we get closest axial slices (z) to the CSC world points

	xxyyzz = world_point
	orientation = np.array(axial_t2_df.ImageOrientationPatient.values.tolist())
	position = np.array(axial_t2_df.ImagePositionPatient.values.tolist())
	ox = orientation[:, :3]
	oy = orientation[:, 3:]
	oz = np.cross(ox, oy)
	t = xxyyzz.reshape(-1, 1, 3) - position.reshape(1, -1, 3)
	dis = (oz.reshape(1, -1, 3) * t).sum(-1)  # np.dot(point-s,oz)
	fdis = np.fabs(dis)
	closest_z = fdis.argmin(-1)
	closest_fdis = fdis.min(-1)
	closest_df = axial_t2_df.iloc[closest_z]

	if 1:
		#<todo> hard/soft assigment, multi/single assigment
		# no assignment based on distance

		# allow point found in multi group
		num_group   = len(axial_t2_df['group'].unique())
		point_group = axial_t2_df.group.values[fdis.argsort(-1)[:, :3]].tolist()
		point_group = [list(set(g)) for g in point_group]
		group_point = [[] for g in range(num_group)]
		for l in range(5):
			for k in point_group[l]:
				group_point[k].append(l)
				# print(k)
				# print(group_point[k])
				# print(group_point)
		group_point = [sorted(list(set(g))) for g in group_point]

	D = len(axial_t2_df)
	assigned_level=np.full(D,fill_value=0, dtype=int)
	for group in range(num_group):
		point_in_this_group = np.array(group_point[group])  # np.where(closest_df['group'] == group)[0]
		slice_in_this_group = np.where(axial_t2_df['group'] == group)[0]
		if len(point_in_this_group) == 0:
			continue # unassigned, level=0

		level = point_in_this_group[fdis[point_in_this_group][:, slice_in_this_group].argmin(0)] + 1
		assigned_level[slice_in_this_group] = level

	# sor =  (fdis.argmin(0)+1)[closest_z]
	# closest_z= [ fdis.argmin(0)+1]
	return assigned_level, closest_z, dis #dis is soft assignment


#########################################################################3
#visualisation
level_color = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
]

def probability_to_rgb(probability):
	_6_,H,W = probability.shape
	rgb = np.zeros((H, W, 3))
	for i in range(1, 6):
		rgb += probability[i].reshape(H, W, 1) * [[level_color[i]]]
	rgb = rgb.astype(np.uint8)
	return rgb


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

def draw_slice(
	ax, df,
	is_slice =True, scolor=[[1,0,0]], salpha=[0.1],
	is_border=True, bcolor=[[1,0,0]], balpha=[0.1],
	is_origin=True, ocolor=[[1,0,0]], oalpha=[0.1],
	is_arrow=True,
):
	df = df.copy()
	df = df.reset_index(drop=True)

	D = len(df)
	if len(scolor)==1: scolor = scolor*D
	if len(salpha)==1: salpha = salpha*D
	if len(bcolor)==1: bcolor = bcolor*D
	if len(balpha)==1: balpha = balpha*D
	if len(ocolor)==1: ocolor = bcolor*D
	if len(oalpha)==1: oalpha = balpha*D


	#for i,d in df.iterrows():
	for i in range(D):
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

		#outline
		if is_slice:
			ax.plot_surface(gx, gy, gz, color=scolor[i], alpha=salpha[i])

		if is_border:
			line = np.stack([p0, p1, p3, p2] )
			ax.plot(line[:,0], line[:,1], zs=line[:,2], color=ocolor[i], alpha=oalpha[i])

		if is_origin:
			ax.scatter([sx], [sy], [sz],   color=ocolor[i], alpha=oalpha[i])

	#check ordering of slice
	if is_arrow :
		sx0, sy0, sz0 = df.iloc[0].ImagePositionPatient
		sx1, sy1, sz1 = df.iloc[-1].ImagePositionPatient
		arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
		a = Arrow3D([sx0, sx1], [sy0, sy1], [sz0, sz1], **arrow_prop_dict)
		ax.add_artist(a)
