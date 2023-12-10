import numpy as np
import cv2

MOV_AVG_LENGTH = 5

class LaneDetector:
    def __init__(self):
        self.mov_avg_left = np.array([])
        self.mov_avg_right = np.array([])
        self.left_fit = np.array([0, 0, 0])
        self.right_fit = np.array([0, 0, 0])

    # CANNY EDGE DETECTION AND COLOR SELECTION
    def color(self, input_image):
        # Convert the image to HLS color space
        hls = cv2.cvtColor(input_image, cv2.COLOR_BGR2HLS)
        
        # Define a white color range using a mask
        lower_white = np.array([0, 160, 10])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(input_image, lower_white, upper_white)
        
        # Apply the mask to the image
        hls_result = cv2.bitwise_and(input_image, input_image, mask=mask)
        
        # Convert the resulting image to grayscale
        gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding and Gaussian blur for edge detection
        _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(threshold, (3, 3), 11)
        canny = cv2.Canny(blur, 40, 60)
        
        return canny

    # REGION OF INTEREST MASKING
    def region_of_interest(self, img):
        mask = np.zeros_like(img)
        img_shape = img.shape
        
        # Define the vertices for the region of interest
        vertices = np.array([[(150, img_shape[0]), (590, 440), (680, 440), (img_shape[1]-20, img_shape[0])]], dtype=np.int32)
        
        # Create a mask using the defined region of interest
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        
        return masked_image

    # PERSPECTIVE TRANSFORMATION (BIRD'S EYE VIEW)
    def warp(self, img, src, dst):
        src = np.float32([src])
        dst = np.float32([dst])
        
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                                dsize=img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)

    # SLIDING WINDOW METHOD FOR LANE DETECTION
    def sliding_window(self, img_w):
        # Select bottom-half of Image for each column
        histogram = np.sum(img_w[img_w.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((img_w, img_w, img_w)) * 255

        # Find the peak of the left and right halves of the histogram
        # starting point for the left and right lines    
        midpoint = img_w.shape[1] // 2
        leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint
        
        # Number of sliding windows, window height, window width, threshold for new window
        nwindows, window_height, margin, minpix = 9, img_w.shape[0] // 9, 100, 50

        # Left-/right lane pixel indices
        left_lane_inds, right_lane_inds = [], []

        leftx, lefty, rightx, righty = np.array([]), np.array([]), np.array([]), np.array([])

        for window in range(nwindows):
            # Set window boundary
            win_y_low, win_y_high = img_w.shape[0] - (window + 1) * window_height, img_w.shape[0] - window * window_height
            win_xleft_low, win_xleft_high = leftx_base - margin, leftx_base + margin
            win_xright_low, win_xright_high = rightx_base - margin, rightx_base + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = np.nonzero(img_w[win_y_low:win_y_high, win_xleft_low:win_xleft_high])
            if good_left_inds[1].size > minpix:
                leftx_base = win_xleft_low + np.mean(good_left_inds[1]).astype(int)
            
            good_right_inds = np.nonzero(img_w[win_y_low:win_y_high, win_xright_low:win_xright_high])
            if good_right_inds[1].size > minpix:
                rightx_base = win_xright_low + np.mean(good_right_inds[1]).astype(int)


            left_lane_inds.append((good_left_inds[0] + win_y_low, good_left_inds[1] + win_xleft_low))
            right_lane_inds.append((good_right_inds[0] + win_y_low, good_right_inds[1] + win_xright_low))

        
        left_lane_inds, right_lane_inds = np.concatenate(left_lane_inds, axis=1), np.concatenate(right_lane_inds, axis=1)
        leftx, lefty, rightx, righty = left_lane_inds[1], left_lane_inds[0], right_lane_inds[1], right_lane_inds[0]

        if len(leftx)==0 or len(lefty)==0:
            left_fit = [0 ,0, 0]
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
        
        if rightx is None or righty is None:
            right_fit = [0, 0, 0]
        else:
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    
    def fit_from_lines(self, left_fit, right_fit, img_w):
        nonzero = img_w.nonzero()
        nonzeroy, nonzerox = nonzero[0], nonzero[1]
        margin = 100

        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
            (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin))
        )

        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
            (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin))
        )

        # Extract left and right line pixel positions
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    # Drawing lane lines on the image and calculating curvature and off-center
    def draw_lines(self, img, img_w, left_fit, right_fit, perspective):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img_w).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        # Generate x values for the polynomial fits
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Recast the x and y points into a usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, [np.int32(pts)], (0, 255, 0))

        # Warp the blank back to the original image space using inverse perspective matrix (Minv)
        newwarp = self.warp(color_warp, perspective[1], perspective[0])

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.2, 0)

        # Create an image to draw the lane lines for visualization purposes
        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
        cv2.polylines(color_warp_lines, np.int32([pts_right]), isClosed=False, color=(0, 255, 0), thickness=25)
        cv2.polylines(color_warp_lines, np.int32([pts_left]), isClosed=False, color=(0, 255, 0), thickness=25)
        newwarp_lines = self.warp(color_warp_lines, perspective[1], perspective[0])

        # Combine the lane lines with the result image
        result = cv2.addWeighted(result, 1, newwarp_lines, 1, 0)

        # Get image height and evaluate the y-coordinate for curvature calculation
        img_height = img.shape[0]
        y_eval = img_height

        # Define conversion factors from pixel to meter
        ym_per_pix = 30 / 720. 
        xm_per_pix = 3.7 / 700 

        # Generate y values for curvature calculation
        ploty = np.linspace(0, img_height - 1, img_height)

        # Fit polynomials in meters for curvature calculation
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

        # Calculate curvature radii for the left and right lanes
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                            (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # Average the curvature radii for the left and right lanes
        radius1 = round((float(left_curverad) + float(right_curverad))/2., 2)

        # Compare first and end x coordinates on the left lane to determine the curve direction
        curve_direction = 'Left Curve' if left_fitx[0] - left_fitx[-1] > 60 else 'Right Curve' if left_fitx[-1] - left_fitx[0] > 60 else 'Straight'
        radius = -5729.57795 / radius1 if curve_direction == 'Left Curve' else 5729.57795 / radius1 if curve_direction == 'Right Curve' else 5729.57795 / radius1

        # Off-center Calculation: if Center = 0, left < 0, right > 0
        # Length with respect to the x-axis
        lane_width = (right_fit[2] - left_fit[2]) * xm_per_pix
        center = (right_fit[2] - left_fit[2]) / 2
        off_left = (center - left_fit[2]) * xm_per_pix
        off_right = -(right_fit[2] - center) * xm_per_pix
        off_center = round((center - img.shape[0] / 2.) * xm_per_pix, 2)

        # Display curvature and off-center information on the image
        text = "Angle = %s [degrees]\noffcenter = %s [m]" % (str(radius), str(off_center))
        for i, line in enumerate(text.split('\n')):
            i = 550 + 20 * i
            cv2.putText(result, line, (0, i), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return result, radius

    def process_frame(self, frame):
        # EDGE DETECTION
        edges = self.color(frame)
        img_b = self.region_of_interest(edges)

        # PERSPECTIVE TRANSFORM: SKY VIEW
        src = [480, 500], [800, 500], [img_b.shape[1] - 50, img_b.shape[0]], [150, img_b.shape[0]]
        line_dst_offset = 200
        dst = [src[3][0] + line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, src[2][1]], \
              [src[3][0] + line_dst_offset, src[3][1]]

        img_w = self.warp(img_b, src, dst)

        try:
            self.left_fit, self.right_fit = self.fit_from_lines(self.left_fit, self.right_fit, img_w)
            self.mov_avg_left = np.append(self.mov_avg_left, np.array([self.left_fit]), axis=0)
            self.mov_avg_right = np.append(self.mov_avg_right, np.array([self.right_fit]), axis=0)

        except Exception:
            self.left_fit, self.right_fit = self.sliding_window(img_w)
            self.mov_avg_left = np.array([self.left_fit])
            self.mov_avg_right = np.array([self.right_fit])

        # Moving average
        self.left_fit = np.array([np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
                                  np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
                                  np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])])
        self.right_fit = np.array([np.mean(self.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
                                   np.mean(self.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
                                   np.mean(self.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])])

        if self.mov_avg_left.shape[0] > 1000:
            self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]
        if self.mov_avg_right.shape[0] > 1000:
            self.mov_avg_right = self.mov_avg_right[0:MOV_AVG_LENGTH]

        final, degrees = self.draw_lines(frame, img_w, self.left_fit, self.right_fit, perspective=[src, dst])

        return final, degrees


def main():
    lane_detector = LaneDetector()

    # Video Capture
    cap = cv2.VideoCapture('./images/LaneToSteeringChallenge.mp4')

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('images/LaneToSteeringChallenge.avi', fourcc, 20.0, (1280, 720))

    while True:
        start = time.time()
        ret, frame = cap.read()
        if ret is True:
            image = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        else:
            print("NO FRAMES")
            break

        final, degrees = lane_detector.process_frame(image)

        out.write(final)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
