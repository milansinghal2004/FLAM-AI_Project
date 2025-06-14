import streamlit as st
from PIL import Image
from rembg import remove
from io import BytesIO
from utils import composite_images, save_image, estimate_light_direction, create_shadow_layer, composite_with_shadow, histogram_matching
import numpy as np
import cv2


def remove_background(image):
    """Removes the background from the given image using RemBG."""
    try:
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        output = remove(img_bytes)
        if output is None:
            raise Exception("RemBG returned None (background removal failed).")
        return Image.open(BytesIO(output))
    except Exception as e:
        print(f"Error during background removal: {e}")
        return None

def adjust_colors(image, temperature, red, green, blue):
    """Adjusts the color temperature and RGB channels of a PIL Image."""
    image_np = np.array(image).astype(np.float32) / 255.0

    # Color Temperature Adjustment
    if temperature > 0:  # Warmer
        image_np[:, :, 0] = np.clip(image_np[:, :, 0] - temperature / 255.0, 0, 1)
    elif temperature < 0:  # Cooler
        image_np[:, :, 2] = np.clip(image_np[:, :, 2] + abs(temperature) / 255.0, 0, 1)

    # RGB Channel Adjustments
    image_np[:, :, 0] = np.clip(image_np[:, :, 0] + red / 255.0, 0, 1)
    image_np[:, :, 1] = np.clip(image_np[:, :, 1] + green / 255.0, 0, 1)
    image_np[:, :, 2] = np.clip(image_np[:, :, 2] + blue / 255.0, 0, 1)

    return Image.fromarray(np.uint8(np.clip(image_np * 255, 0, 255)))


def main():
    st.title("SceneBlend AI")
    st.write("")

    st.header("1. Upload Images")
    person_image_file = st.file_uploader("Upload Person Image (PNG or JPG)", type=["png", "jpg", "jpeg"])
    background_image_file = st.file_uploader("Upload Background Image (PNG or JPG)", type=["png", "jpg", "jpeg"])

    if person_image_file is not None and background_image_file is not None:
        try:
            person_image = Image.open(person_image_file)
            background_image = Image.open(background_image_file)

            st.subheader("Input Images")
            col1, col2 = st.columns(2)
            with col1:
                st.image(person_image, caption="Original Person Image", use_column_width=True)
            with col2:
                st.image(background_image, caption="Background Image", use_column_width=True)

            st.header("2. Processing")
            with st.spinner("Removing background from person image..."):
                person_image_no_bg = remove_background(person_image)

            if person_image_no_bg is None:
                st.error("Background removal failed. Please try a different image or check RemBG.")
                st.stop()

            st.success("Background removed!")

            with st.spinner("Matching person image colors to background..."):
                background_np = np.array(background_image)
                person_no_bg_np = np.array(person_image_no_bg.convert('RGB'))  # Histogram matching works on 3-channel images
                person_image_matched = Image.fromarray(histogram_matching(person_no_bg_np, background_np))
            st.success("Colors matched!")

            st.subheader("Person Image (No Background, Color Matched)")
            st.image(person_image_matched, caption="Person Image (No Background, Color Matched)", use_column_width=True)

            st.header("3. Compositing Settings")
            with st.sidebar:
                st.subheader("Positioning")
                x_offset = st.slider("X Offset", -background_image.width, background_image.width, 0)
                y_offset = st.slider("Y Offset", -background_image.height, background_image.height, 0)
                scale = st.slider("Scale", 0.1, 2.0, 1.0, 0.05)

                st.write("")
                st.subheader("Shadow Settings")
                shadow_distance = st.slider("Shadow Distance", 0, 100, 20)
                shadow_blur = st.slider("Shadow Blur", 0, 51, 10)  # Increased max

                st.write("")
                st.subheader("Color Adjustments")  # Updated section title
                color_temp = st.slider("Color Temperature", -100, 100, 0)
                red_adjust = st.slider("Red Adjustment", -100, 100, 0)  # New slider
                green_adjust = st.slider("Green Adjustment", -100, 100, 0)  # New slider
                blue_adjust = st.slider("Blue Adjustment", -100, 100, 0)  # New slider

            with st.spinner("Adjusting Colors..."):  # Changed spinner message
                person_image_adjusted = adjust_colors(person_image_matched, color_temp, red_adjust, green_adjust, blue_adjust) # Call adjust_colors
                st.success("Colors adjusted!")

            st.subheader("Person Image (No Background, Color Matched, Adjusted)")
            st.image(person_image_adjusted, caption="Person Image (No Background, Color Matched, Adjusted)", use_column_width=True) # Updated caption

            with st.spinner("Estimating light direction..."):
                light_direction = estimate_light_direction(background_image)
                st.success(f"Light direction estimated: {light_direction}")

            with st.spinner("Creating shadow layer..."):
                shadow_layer = create_shadow_layer(person_image_no_bg, light_direction, shadow_distance, shadow_blur)
                st.success("Shadow layer created!")

            with st.spinner("Compositing images with shadow..."):
                composite = composite_with_shadow(background_image.copy(), person_image_adjusted, shadow_layer)  # Use adjusted image!
                composite = composite_images(composite, person_image_adjusted, x_offset, y_offset, scale)  # Use adjusted image!
                if composite:
                    st.success("Images composited!")
                    st.header("4. Final Composite")
                    st.image(composite, caption="Composite Image with Shadow", use_column_width=True)

                    image_bytes = BytesIO()
                    composite.save(image_bytes, format="PNG")
                    image_bytes = image_bytes.getvalue()
                    st.download_button(
                        label="Download Composite Image",
                        data=image_bytes,
                        file_name="composite.png",
                        mime="image/png",
                    )
                else:
                    st.error("Failed to composite images. Check the error messages.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()