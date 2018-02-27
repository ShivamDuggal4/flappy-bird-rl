import numpy as np
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer


def process_image(image,image_rows,image_columns):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image,(image_rows,image_columns))
	#image = skimage.exposure.rescale_intensity(image,out_range=(0,255))
	
	return image


def form_batches(states,action_values,image_rows,image_columns,image_channels):
	return np.array(states).reshape(-1,image_rows,image_columns,image_channels),np.array(action_values).reshape(-1)

def form_cumulative_state(cumulative_state,image_rows,image_columns,image_channels):
	final_states = []
	state1 = process_image(cumulative_state[0],image_rows,image_columns)
	state2 = process_image(cumulative_state[1],image_rows,image_columns)
	state3 = process_image(cumulative_state[2],image_rows,image_columns)
	state4 = process_image(cumulative_state[3],image_rows,image_columns)
	final_states.append(np.stack((state4,state3,state2,state1), axis=2))
	return np.array(final_states).reshape(-1,image_rows,image_columns,image_channels)

