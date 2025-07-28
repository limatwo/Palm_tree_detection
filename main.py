import cv2
import numpy as np

# Read and resize image
img = cv2.imread("tree.png")  
img = cv2.resize(img, (640, 480))

# Step 1: Convert to Grayscale (Reduce Computational Cost)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Step 1: Grayscale", gray)
cv2.imwrite("results/step1_grayscale.png", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Apply Gaussian Blur (Smooth Edges, Reduce Noise)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow("Step 2: Blurred", blurred)
cv2.imwrite("results/step2_blurred.png", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Detect Edges of Leaves (Edge Detection)
edges = cv2.Canny(blurred, 20, 80)  # Detects edges
cv2.imshow("Step 3: Edge Detection", edges)
cv2.imwrite("results/step3_edges.png", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4: Remove Noise (Morphological Closing to Fill Gaps)
kernel = np.ones((9, 9), np.uint8)
cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # Fills small gaps
cv2.imshow("Step 4: Noise Removed", cleaned_edges)
cv2.imwrite("results/step4_noise_removed.png", cleaned_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Step 5: Find Contours and Determine Centroid
contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    # Get the largest contour (assuming it's the tree)
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute centroid using image moments
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  # X coordinate of centroid
        cy = int(M["m01"] / M["m00"])  # Y coordinate of centroid
    else:
        cx, cy = 0, 0  # Default to (0,0) if centroid calculation fails

    # Draw contours and centroid
    output = img.copy()
    cv2.drawContours(output, [largest_contour], -1, (0, 255, 0), 4)  # Green contour
    cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)  # Red centroid

    # # Compute image center
    # h, w = img.shape[:2]
    # img_center = (w // 2, h // 2)

    # # Draw image center (blue)
    # cv2.circle(output, img_center, 10, (255, 0, 0), -1)  

    # Show final result
    cv2.imshow("Step 5: Tree Contour & Centroid", output)
    cv2.imwrite("results/step5_final.png", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours detected!")