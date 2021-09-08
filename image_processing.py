import os
import cv2
import numpy as np
import pandas as pd
import math

from HSVFilter import HSVFilter
from circle_fit import circle_fit_by_taubin

directory = os.path.abspath("./Data/Pictures/")
avg_dist = 0
avg_dist_n = 0
three_markers = False


def get_distance(pt1: (int, int), pt2: (int, int)):
    a = pt2[0] - pt1[0]
    b = pt2[1] - pt1[1]
    return math.sqrt(a**2 + b**2)


def double_area(a: (int, int), b: (int, int), c: (int, int)):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def menger_curvature(a: (int, int), b: (int, int), c: (int, int)):
    side_length1 = get_distance(a, b)
    side_length2 = get_distance(b, c)
    side_length3 = get_distance(c, a)
    return float((2 * abs(double_area(a, b, c))) / (side_length1 * side_length2 * side_length3))


def get_mid_point(coords_X: list, coords_Y: list, size: int):
    if size % 2 == 0:
        mid_x = round((coords_X[int(size/2)] + coords_X[int(size/2 - 1)]) / 2)
        mid_y = round((coords_Y[int(size/2)] + coords_Y[int(size/2 - 1)]) / 2)
        return mid_x, mid_y
    else:
        mid_x = coords_X[int(size/2)]
        mid_y = coords_Y[int(size/2)]
        return mid_x, mid_y


def update_avg_dist(new_dist):
    global avg_dist, avg_dist_n
    avg_dist = (new_dist + avg_dist_n * avg_dist) / (avg_dist_n + 1)
    avg_dist_n += 1


def get_min_distance_cont_index(coords, remaining_conts, remaining_coords):
    min_dist = None
    min_index = -1
    for i in range(0, len(remaining_conts)):
        dist = get_distance(coords, remaining_coords[i])
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index


def sort_contours(conts, sort_by_distance=False, sort_by_top_left=False):
    if sort_by_top_left:
        bounding_boxes = [cv2.boundingRect(cont) for cont in conts]
        conts, bounding_boxes = [list(y) for y in zip(*sorted(zip(conts, bounding_boxes), key=lambda b: -b[1][0]+b[1][1]))]
    if sort_by_distance:
        coords = []
        for cont in conts:
            moment = cv2.moments(cont)
            coords.append((int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"])))
        for j in range(len(conts) - 1):
            distance = get_distance(coords[j], coords[j + 1])
            if distance > 75:
                offset = get_min_distance_cont_index(coords[j], conts[j + 2:], coords[j + 2:])
                if offset != -1:
                    next_cnt_index = j + 2 + offset
                    conts[j + 1], conts[next_cnt_index] = conts[next_cnt_index], conts[j + 1]
                    coords[j+1], coords[next_cnt_index] = coords[next_cnt_index], coords[j+1]
                    new_distance = get_distance(coords[j], coords[j + 1])
                    update_avg_dist(new_distance)
    return conts


def get_curvature_and_positional_data(three_markers: bool) -> (pd.DataFrame, pd.DataFrame):
    curvature_d, positional_d = [], []
    for exp_date in os.listdir(directory):
        if not(exp_date.startswith("23-07-21")):
            continue
        exp_date_path = os.path.join(directory, exp_date)
        for exp_num in os.listdir(exp_date_path):
            exp_path = os.path.join(exp_date_path, exp_num)
            for image_name in os.listdir(exp_path):
                img_path = os.path.join(exp_path, image_name)

                # Load the image
                img = cv2.imread(img_path)

                x_margin = 600
                y_margin = 200
                img = img[y_margin:-y_margin, x_margin:-x_margin]

                # Apply contrast
                alpha = 1.1  # Contrast control (1.0 - 3.0)
                beta = -30  # Brightness control (0 - 100)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

                # Set image size = 1/2 * original size
                height, width, _ = img.shape
                new_height, new_width = int(height / 2), int(width / 2)
                img = cv2.resize(img, (new_width, new_height))

                # Opens HSVFilter color picker to help fine tune the threshold values
                images_to_adjust = []
                if image_name in images_to_adjust:
                    window = HSVFilter(img)
                    window.show()
                    cv2.destroyAllWindows()

                # Set lower and upper range for the green marker
                lower_range = np.array([65, 50, 150])
                upper_range = np.array([88, 255, 255])

                # Get only the marker areas from the image using a color threshold
                mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_range, upper_range)

                # Dilate mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

                # Debugging
                if image_name in images_to_adjust:
                    img[mask == 255] = (0, 0, 255)
                    cv2.imshow(f"{image_name}", img)
                    cv2.waitKey(0)

                # Find contours in the thresholded image
                cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = list(filter(lambda cnt: cv2.contourArea(cnt) > 100, cnts))
                cnts = sort_contours(cnts, sort_by_distance=True, sort_by_top_left=True)

                # Get all marker areas as points
                blank_img = np.zeros((new_height, new_width), np.uint8)
                i = 0
                d = []
                for c in cnts:
                    # compute the center of the contour
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # fill the data with the coords
                    d.append({'Chamber': i, 'X-Value': cX, 'Y-Value': cY})
                    # draw the center of the shape on the new image
                    cv2.circle(blank_img, (cX, cY), 3, (255, 255, 255), -1)
                    # cv2.putText(blank_img, f"chamber {i}: {(cX, cY)}", (cX + 5, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
                    i += 1

                # Debugging
                if image_name in images_to_adjust:
                    img[blank_img == 255] = (0, 0, 255)
                    cv2.imshow(f"{image_name}", img)
                    cv2.waitKey(0)

                # Create dataframe of coordinates using all marker points
                df = pd.DataFrame(d)

                # Try to fit a circle to the points
                # all_points = df.drop('Chamber', axis=1).values[1:]
                # fitted_circle = circle_fit_by_taubin(all_points)
                # cv2.circle(blank_img, (int(fitted_circle[0][0]), int(fitted_circle[0][1])), int(fitted_circle[1]), (255,255, 255), 1)

                # Extract X and Y coordinates
                coords_X, coords_Y = df['X-Value'].tolist(), df['Y-Value'].tolist()
                print(f"{exp_date}/{exp_num}/{image_name} - ", coords_X, coords_Y)

                # Calculate the curvatures
                curvatures = []
                if three_markers:
                    a = (coords_X[0], coords_Y[0])
                    b = get_mid_point(coords_X, coords_Y, len(coords_X))
                    c = (coords_X[-1], coords_Y[-1])
                    curvatures.append(menger_curvature(a, b, c))

                    blank_img = np.zeros((new_height, new_width, 3), np.uint8)
                    cv2.putText(blank_img, f"curvature: {menger_curvature(a, b, c)}",
                                (b[0] - 100, b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
                    for x, y in (a, b, c):
                        cv2.circle(blank_img, (x, y), 1, (255, 255, 255), -1)
                else:
                    for i in range(0, 12, 3):
                        a = (coords_X[i], coords_Y[i])
                        b = (coords_X[i + 1], coords_Y[i + 1])
                        c = (coords_X[i + 2], coords_Y[i + 2])
                        curvatures.append(menger_curvature(a, b, c))

                # Debugging:
                # img = cv2.addWeighted(img, 0.5, blank_img, 1, 0)
                # cv2.imshow(f"{image_name}", img)
                # cv2.waitKey(0)

                # Fill the dataframes with the extracted positional and curvature data
                curvature_d.append({
                    "image": image_name,
                    **{f"curvature {i+1}": curvatures[i] for i in range(len(curvatures))}
                })

                chambers = [0, 6, 11] if three_markers else range(2, 12)
                positional_d.append({
                    "image": image_name,
                    ** {f"chamber {i+1} X": coords_X[i] - coords_X[0] for i in chambers},
                    ** {f"chamber {i+1} Y": coords_Y[i] - coords_Y[0] for i in chambers}
                })
    return pd.DataFrame(curvature_d), pd.DataFrame(positional_d)


if __name__ == '__main__':
    curvature_data, positional_data = get_curvature_and_positional_data(three_markers)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
        print("Curvature data:")
        print(curvature_data)
        print("Positional data:")
        print(positional_data)
    cv2.destroyAllWindows()
