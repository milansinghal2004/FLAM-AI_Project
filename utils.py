from PIL import Image
import cv2
import numpy as np

def composite_images(background_image, person_image, x_offset, y_offset, scale):
    """Composites the person image onto the background image."""
    person_width, person_height = person_image.size
    new_person_width = int(person_width * scale)
    new_person_height = int(person_height * scale)

    person_image = person_image.resize((new_person_width, new_person_height), Image.LANCZOS)

    # Make sure the person image is RGBA
    if person_image.mode != "RGBA":
        person_image = person_image.convert("RGBA")

    # Ensure background is RGBA (important for alpha compositing!)
    if background_image.mode != "RGBA":
        background_image = background_image.convert("RGBA")

    background_width, background_height = background_image.size
    # Paste the person onto the background
    try:
        background_image.paste(person_image, (x_offset, y_offset), person_image)
    except ValueError as e:
        print(f"Error during paste: {e}")  # Print the error for debugging
        return None

    return background_image

def save_image(image, filename):
    """Saves the image to the specified filename."""
    image.save(filename)

def estimate_light_direction(image):
    """Estimates the light direction from a grayscale image (very basic)."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    avg_grad_x = np.mean(grad_x)
    avg_grad_y = np.mean(grad_y)

    # Light direction is opposite to the gradient direction
    light_direction = (-avg_grad_x, -avg_grad_y)
    
    # Normalize the light direction vector
    magnitude = np.sqrt(light_direction[0]**2 + light_direction[1]**2)
    if magnitude > 0:
        light_direction = (light_direction[0] / magnitude, light_direction[1] / magnitude)
    else:
        light_direction = (0, -1)  # Default to downwards if no gradient

    return light_direction

def create_shadow_layer(person_image_no_bg, light_direction, shadow_distance, shadow_blur):
    """Creates a synthetic shadow layer."""
    person_np = np.array(person_image_no_bg)
    alpha = person_np[:, :, 3] / 255.0  # Alpha channel
    shadow = alpha * 1.0  # Shadow intensity

    # Shift the shadow
    dx, dy = light_direction
    dx *= shadow_distance
    dy *= shadow_distance
    
    rows, cols = shadow.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow_shifted = cv2.warpAffine(shadow, M, (cols, rows)) # Shift the shadow

    # Blur the shadow
    if shadow_blur > 0:  # Blur only when blur value is more than zero to avoid errors.
        # Ensure shadow_blur is odd
        if shadow_blur % 2 == 0:
            shadow_blur += 1  # Make odd for GaussianBlur
        print(f"Shadow blur value: {shadow_blur}")  # DEBUG PRINT
        shadow_blurred = cv2.GaussianBlur(shadow_shifted, (shadow_blur, shadow_blur), 0)
    else:
        shadow_blurred = shadow_shifted  # No blur

    shadow_layer = np.zeros((rows, cols, 4), dtype=np.float32)  # RGBA shadow layer
    shadow_layer[:, :, 3] = shadow_blurred  # Alpha channel of the shadow

    return shadow_layer

def composite_with_shadow(background, person, shadow):
    """Composites person and shadow onto the background (OpenCV)."""
    background_np = np.array(background).astype(np.float32) / 255.0
    person_np = np.array(person).astype(np.float32) / 255.0
    shadow = shadow.astype(np.float32)
    
    # RESIZE SHADOW TO BACKGROUND DIMENSIONS
    shadow = cv2.resize(shadow, (background_np.shape[1], background_np.shape[0])) # Resize shadow to match background dimensions

    h, w = shadow.shape[:2]

    # Resize person to be same dimensions as shadow
    person = cv2.resize(person_np, (w, h))
    # Ensure person has 4 channels by converting to RGBA
    if person.shape[2] == 3:
        person = cv2.cvtColor(person, cv2.COLOR_RGB2RGBA) #Force 4 channel by converting to RGBA format
    #Ensure that the shadow is four channels
    if len(shadow.shape) == 2: #if it is 2 channels then convert to GRAY2RGBA
         shadow =  cv2.cvtColor(shadow, cv2.COLOR_GRAY2RGBA)
    elif shadow.shape[2] == 3: #else RGB2RGBA
        shadow =  cv2.cvtColor(shadow, cv2.COLOR_RGB2RGBA)

    alpha = person[:,:,3] # Index 3 may be out of bounds
    alpha_inv = 1 - alpha

    # Composite shadow with background
    for c in range(0, 3):
        background_np[:, :, c] = background_np[:, :, c] * (1 - shadow[:,:,3]) # Shadow might require 4 channel as well

    for c in range(0,3):
        background_np[:,:,c] += shadow[:,:,3] * 0 #Shadow color is set as black

    # Composite person with background with shadow
    for c in range(0, 3):
        background_np[:, :, c] = (alpha * person[:, :, c] + alpha_inv * background_np[:, :, c])
    
    background_np = np.clip(background_np, 0, 1) # Ensure pixel values are within the range 0 and 1
    return Image.fromarray(np.uint8(background_np * 255))

from PIL import Image
import cv2
import numpy as np

# (Previous functions: composite_images, save_image, estimate_light_direction,
#  create_shadow_layer, composite_with_shadow)

def histogram_matching(source, template):
    """Matches the histogram of the source image to the template image."""
    old_shape = source.shape
    source = source.ravel()
    template = template.ravel()
    # Get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Calculate s_cdf
    s_cdf = np.cumsum(s_counts).astype(np.float64)
    s_cdf /= s_cdf[-1]

    # Calculate t_cdf
    t_cdf = np.cumsum(t_counts).astype(np.float64)
    t_cdf /= t_cdf[-1]

    # We need to interpolate the s_cdf value
    interp_s_values = np.interp(s_cdf, t_cdf, t_values)
    return interp_s_values[bin_idx].reshape(old_shape).astype(np.uint8)