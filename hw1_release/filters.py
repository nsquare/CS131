import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    # can be done using two loops and np.dot
    # but lets do it in more elaborate way
    
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    pad_size = Hk//2 # Assuming kernel is square
    out = np.zeros((Hi, Wi))
    kernel_copy = np.flip(kernel,1)
    kernel_copy = np.flip(kernel_copy,0)
    for i in range(Hi):
        for j in range(Wi):
            val =0
            for k in range(Hk):
                for l in range(Wk):
                    if (i+k-pad_size  >-1) and (j+l - pad_size) >-1 and (i+k-pad_size <Hi)  and (j+l - pad_size)< Wi:
                        elem = image[i+k-pad_size,j+l-pad_size]
                    else:
                        elem = 0 
                    val += elem*kernel_copy[k,l]     
            out[i,j]= val
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    kernel_copy = np.flip(kernel,1)
    kernel_copy = np.flip(kernel_copy,0)
    out = np.zeros((Hi, Wi))
    zero_pad_im = zero_pad(image,Hk//2,Wk//2)
    
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = np.sum(zero_pad_im[i:i+Hk,j:j+Wk]*kernel_copy)
    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    kernel_copy = np.flip(kernel,1)
    kernel_copy = np.flip(kernel_copy,0)
    out = np.zeros((Hi, Wi))
    zero_pad_im = zero_pad(image,Hk//2,Wk//2)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = np.dot(zero_pad_im[i:i+Hk,j:j+Wk].reshape(-1),kernel_copy.reshape(-1))
    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    g_copy = np.flip(np.flip(g,0),1)
    out = conv_faster(f, g_copy)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
        
    """

    g_copy = np.flip(np.flip(g,0),1)
    g_mean = np.mean(g_copy)
    g_copy = g_copy - g_mean
    out = conv_faster(f, g_copy)
    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
        
      
    """
    
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    zero_pad_f = zero_pad(f,Hk//2,Wk//2)
    g_copy = (g-np.mean(g))/np.std(g)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = np.dot((zero_pad_f[i:i+Hk,j:j+Wk].reshape(-1)-np.mean(zero_pad_f[i:i+Hk,j:j+Wk]))/np.std(zero_pad_f[i:i+Hk,j:j+Wk]) ,g.reshape(-1))
    

    return out
