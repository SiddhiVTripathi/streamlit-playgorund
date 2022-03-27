from tools import *
import functools
import os
import streamlit as st
from matplotlib import gridspec
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import sys
from PIL import Image
import numpy as np
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#Converting Tensor to image
def tensor_to_image(tensor):
    tensor =tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


#loading images 
def load_img(img):
    max_dim = 256
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs,extractor):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


@tf.function()
def train_step(image, extractor, weight=None):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs,extractor)
    if weight != None:
        loss+=weight*tf.image.total_variation(image) #weight is total variation weight
  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

def read_tensor_from_image(image, input_height=256, input_width=256,
            input_mean=0, input_std=255):

  float_caster = tf.cast(image, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize(dims_expander, [input_height, input_width])

  return resized






# = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
#hub_module = hub.load(hub_handle)

col1,col2 = st.columns(2)
col1.header("Image")
col2.header("Style")
with st.form('app'):
    with col1:
        contentFile = st.file_uploader("upload content image", type=["png","jpg","jpeg"],key=1)
        if contentFile:
            contentImg = load_image_pil(contentFile)
            st.image(contentImg, use_column_width=True)
        content_weight = st.slider('Adjust your Content intensity',1e2,1e5)    
    with col2:
        styleFile = st.file_uploader("upload style image", type=["png","jpg","jpeg"],key=2)
        if styleFile:
            styleImg = load_image_pil(styleFile)
            st.image(styleImg, use_column_width=True)
        style_weight = st.slider('Adjust your style intensity',1e-6,1e-1)
    
    submit = st.form_submit_button('Tranfer!')

if submit and contentImg and styleImg:
    style_image = load_img(np.asarray(styleImg))
    content_image = load_img(np.asarray(contentImg))
    content_layers = ['block5_conv2'] 

    style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)

    extractor = StyleContentModel(style_layers, content_layers)

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.03, beta_1=0.99, epsilon=1e-1)

    start = time.time()

    epochs = 10
    steps_per_epoch = 50

    step = 0
    bar = st.progress(0)
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            bar.progress(int(step/(epochs*steps_per_epoch)*100))
            train_step(image,extractor=extractor, weight=30)
            print(".", end='')
        im = tensor_to_image(image)
        # im.save("output/transfered_{}.jpg".format(step))
        print("Train step: {}".format(step))
        end = time.time()
        print(" Total time: {:.1f}".format(end-start))
    # im.show()
    st.image(im,caption="Stylized Image")
