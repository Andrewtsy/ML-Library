import numpy as np

def padding(p,A):
  return np.pad(A,((0,0),(p,p),(p,p),(0,0)))

def conv(W,s,A):
  f = W.shape[0]
  n_e = A.shape[0]
  n_h = (A.shape[1]-f)//s + 1
  n_w = (A.shape[2]-f)//s + 1
  n_c = W.shape[3]
  A_conv_W = np.zeros((n_e,n_h,n_w,n_c))
  for i_e in range(n_e):
    for i_h in range(n_h):
      for i_w in range(n_w):
        for i_c in range(n_c):
          A_conv_W[i_e,i_h,i_w,i_c] = np.sum(A[i_e,i_h*s:i_h*s+f,i_w*s:i_w*s+f,:]*W[:,:,:,i_c])
  return A_conv_W  

def pooling(f,s,A):
  n_e = A.shape[0]
  n_h = (A.shape[1]-f)//s + 1
  n_w = (A.shape[2]-f)//s + 1
  n_c = A.shape[3]
  A_pool = np.zeros((n_e,n_h,n_w,n_c))
  Z = []
  for i_e in range(n_e):
    for i_h in range(n_h):
      for i_w in range(n_w):
        for i_c in range(n_c):
          x = np.argmax(A[i_e,i_h*s:i_h*s+f,i_w*s:i_w*s+f,i_c])
          i_w_max = x%f + i_w*s
          i_h_max = x//f + i_h*s
          Z.append((i_e,i_h_max,i_w_max,i_c))
          A_pool[i_e,i_w,i_h,i_c] = A[i_e,i_w_max,i_h_max,i_c]
  return A_pool, Z  

def convolutional(b,W,s,p,activ,X):
  Z = conv(W,s,padding(p,X))+b
  A = h(activ,Z)
  return A, Z

def DJ_DA_L(y, A_L):
    return -(y / A_L) / y.shape[0]

def DJ_DA_prev(act, b, W, Z, A, A_prev, DJ_DA):
    DJ_DZ = DJ_DA * h_derivative(act, Z)
    DJ_Db = np.sum(DJ_DZ, axis=0)
    DJ_DW = np.matmul(A_prev.T, DJ_DZ)
    DJ_DA_prev = np.matmul(DJ_DZ, W.T)
    return DJ_Db, DJ_DW, DJ_DA_prev

def DJ_DZ_L(y, A_L):
    (A_L - y) / y.shape[0]

def DJ_DW_conv(W_shape,DJ_DZ,A_prev_pad,s) 
  DJ_DW = np.zeros(W_shape)
  for i1 in range(W_shape[0]):
    ind_1 = np.arange(i1,(DJ_DZ.shape[1]-1)*s+i1+1,s)
    for i2 in range(W_shape[1]):
      ind_2 = np.arange(i2,(DJ_DZ.shape[2]-1)*s+i2+1,s)
      for i3 in range(W_shape[2]):
        A_res = A_prev_pad[:,:,:,i3]
        A_res = A_res[:,:,ind_2]
        A_res = A_res[:,ind_1,:]
        for i4 in range(W_shape[3]):
          DJ_DW[i1,i2,i3,i4] = np.sum(DJ_DZ[:,:,:,i4]*A_res)
  return DJ_DW    

def DJ_DA_prev_conv(A_prev_shape,DJ_DZ,W,p,s):
  f = W.shape[0]
  DJ_DA_prev = np.zeros(A_prev_shape)
  for i1 in range(A_prev_shape[0]):
    for i2 in range(A_prev_shape[1]):
      a2_min = max((i2+p-f)//s+1,0)
      a2_max = min((i2+p)//s,DJ_DZ.shape[1]-1)
      ind_0 = np.arange(i2+p-a2_min*s,i2+p-a2_max*s-1,-s)
      for i3 in range(A_prev_shape[2]):
        a3_min = max((i3+p-f)//s+1,0)
        a3_max = min((i3+p)//s,DJ_DZ.shape[2]-1)
        ind_1 = np.arange(i3+p-a3_min*s,i3+p-a3_max*s-1,-s)
        DJ_DZ_res = DJ_DZ[i1,:,:,:]
        DJ_DZ_res = DJ_DZ_res[:,a3_min:a3_max+1,:]
        DJ_DZ_res = DJ_DZ_res[a2_min:a2_max+1,:,:]
        for i4 in range(A_prev_shape[3]):
          W_res = W[:,:,i4,:]
          W_res = W_res[:,ind_1,:]
          W_res = W_res[ind_0,:,:]
          DJ_DA_prev[i1,i2,i3,i4] = np.sum(DJ_DZ_res*W_res)
  return DJ_DA_prev

def gradients_one_layer_convolutional(l,activ,p,s,b,W,Z,A,A_prev,DJ_DA):
  A_prev_pad = padding(p,A_prev)
  DJ_DZ = DJ_DA*h_p(activ,Z)
  DJ_Db = np.sum(DJ_DZ,axis=(0,1,2))
  DJ_DW = DJ_DW_conv(W.shape,DJ_DZ,A_prev_pad,s)
  DJ_DA_prev = None
  if l != 1:
    DJ_DA_prev = DJ_DA_prev_conv(A_prev.shape,DJ_DZ,W,p,s)
  return DJ_Db, DJ_DW, DJ_DA_prev

def gradients_one_layer_pooling(l,Z,A,A_prev,DJ_DA):
  n_e = A.shape[0]
  n_h = A.shape[1]
  n_w = A.shape[2]
  n_c = A.shape[3]
  DJ_DA_prev = np.zeros(A_prev.shape)
  if l == 1:
    return None, None, DJ_DA_prev
  i = 0
  for i_e in range(n_e):
    for i_h in range(n_h):
      for i_w in range(n_w):
        for i_c in range(n_c):
          DJ_DA_prev[Z[i]] = DJ_DA_prev[Z[i]] + DJ_DA[i_e,i_h,i_w,i_c] 
          i = i+1 
  return None, None, DJ_DA_prev

def gradients_all_layers(W,b,activation,layer_type,strides,paddings,first_dense,L,A,Z,Y):
  DJ_DA = [DJ_DA_L(Y,A[-1])]
  DJ_Db = []
  DJ_DW = []
  l = L
  while l > 0:
    DJ_Db_l, DJ_DW_l, DJ_DA_l_1 = gradients_one_layer(l,layer_type[l],activation[l],paddings[l],strides[l],b[l],W[l],Z[l],A[l],A[l-1],DJ_DA[0],Y,first_dense):
    if l == first_dense:
      DJ_DA_l_1 = DJ_DA_l_1.reshape(A[l-1].shape)    
    DJ_DA.insert(0,DJ_DA_l_1)
    DJ_Db.insert(0,DJ_Db_l)
    DJ_DW.insert(0,DJ_DW_l)
    l = l-1
  DJ_Db.insert(0,None)
  DJ_DW.insert(0,None)  
  return DJ_Db, DJ_DW  

def initialize_a_W_and_b(layer_type,shape_input,shape_output,filter_size,need_to_flatten):
  if need_to_flatten:
    s_i_f = shape_input[0]*shape_input[1]*shape_input[2]
    W = np.random.randn(s_i_f,shape_output)/np.sqrt(s_i_f)
    b = np.zeros(shape_output)
    return W, b 
  if layer_type == 'dense_layer':
    W = np.random.randn(shape_input,shape_output)/np.sqrt(shape_input)
    b = np.zeros(shape_output)
    return W, b
  if layer_type == 'convolutional_layer':
    s_i_f = shape_input[0]*shape_input[1]*shape_input[2]
    W = np.random.randn(filter_size,filter_size,shape_input[2],shape_output[2])/np.sqrt(s_i_f)
    b = np.zeros(shape_output[2])
    return W, b
  return None, None  

def initialize_all_W_and_b(layer_type,shape_data,L,filters_sizes,first_dense):
  W = [None]
  b = [None,]
  for l in range(1,L+1):
    need_to_flatten = (l == first_dense)
    W_l, b_l = initialize_a_W_and_b(layer_type[l],shape_data[l-1],shape_data[l],filters_sizes[l],need_to_flatten)
    W.append(W_l)
    b.append(b_l)
  return W, b 

def one_step(W,b,activation,layer_type,filters_sizes,strides,paddings,first_dense,L,X,Y,c,la):
  n_e = X.shape[0]
  A, Z = data_in_layers(W,b,activation,layer_type,filters_sizes,strides,paddings,first_dense,L,X)
  error = error_cross_entropy(Y,A[-1])
  DJ_Db, DJ_DW = gradients_all_layers(W,b,activation,layer_type,strides,paddings,first_dense,L,A,Z,Y)
  for l in range(1,L+1):
    if layer_type[l] != 'pooling_layer':
      b[l] = b[l] - c*DJ_Db[l] - 2*la[l]*b[l]/n_e 
      W[l] = W[l] - c*DJ_DW[l] - 2*la[l]*W[l]/n_e
  return b, W, error  

def gradient_descent(shape_data,activation,layer_type,filters_sizes,strides,paddings,first_dense,L,X,Y,c,n_steps,la):
  W, b = initialize_all_W_and_b(layer_type,shape_data,L,filters_sizes,first_dense)
  J = np.array([])
  for i in range(n_steps):
    b, W, error = one_step(W,b,activation,layer_type,filters_sizes,strides,paddings,first_dense,L,X,Y,c,la)
    J = np.append(J,error)
  return b, W, J

def forward_prop(W,b,activation,layer_type,filters_sizes,strides,paddings,first_dense,X):
  A = [X]
  
  for l in range(1,L+1):
    A_prev = A[l-1]
    if l == 1:
      A_prev = A_prev.reshape((A_prev.shape[0],-1))
    A.append(layer(W[l],b[l],activation[l],layer_type[l],filters_sizes[l],strides[l],paddings[l],A_prev))
  return A

def predictions(W,b,activation,layer_type,filters_sizes,strides,paddings,first_dense,X):
  A = data_in_layers(W,b,activation,layer_type,filters_sizes,strides,paddings,first_dense,X)
  return A[-1]  