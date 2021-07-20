import os
import cv2
import numpy as np
import pandas as pd
import math

from circle_fit import circle_fit_by_taubin

directory = os.path.abspath("./Data/Pictures/")
avg_dist = 0
avg_dist_n = 0


def get_min_distance_cnt_index(coords, remaining_cnts, remaining_coords):
    min_dist = None
    min_index = -1
    for i in range(0, len(remaining_cnts)):
        dist = get_distance(coords, remaining_coords[i])
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index


def get_distance(pt1: (int, int), pt2: (int, int)):
    """Returns the Euclidean distance between this point and that point.
    :param pt1: (x: int, y: int)
    :param pt2: (x: int, y: int)
    :return: the distance between pt1 and pt2
    """
    a = pt2[0] - pt1[0]
    b = pt2[1] - pt1[1]
    return math.sqrt(a**2 + b**2)


def double_area(a: (int, int), b: (int, int), c: (int, int)):
    """Returns twice the signed area of the triangle a-b-c.
    :param a: ((int, int)): first point
    :param b: ((int, int)): second point
    :param c: ((int, int)): third point
    :return: Twice the signed area of the triangle a-b-c
    """
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def menger_curvature(a: (int, int), b: (int, int), c: (int, int)):
    side_length1 = get_distance(a, b)
    side_length2 = get_distance(b, c)
    side_length3 = get_distance(c, a)
    return (2 * abs(double_area(a, b, c))) / (side_length1 * side_length2 * side_length3)


def update_avg_dist(new_dist):
    global avg_dist, avg_dist_n
    avg_dist = (new_dist + avg_dist_n * avg_dist) / (avg_dist_n + 1)
    avg_dist_n += 1


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
                offset = get_min_distance_cnt_index(coords[j], conts[j + 2:], coords[j + 2:])
                if offset != -1:
                    next_cnt_index = j + 2 + offset
                    conts[j + 1], conts[next_cnt_index] = conts[next_cnt_index], conts[j + 1]
                    coords[j+1], coords[next_cnt_index] = coords[next_cnt_index], coords[j+1]
                    new_distance = get_distance(coords[j], coords[j + 1])
                    update_avg_dist(new_distance)
    return conts


def get_mid_point(coords_X: list, coords_Y: list, size: int):
    if size % 2 == 0:
        mid_x = round((coords_X[int(size/2)] + coords_X[int(size/2 - 1)]) / 2)
        mid_y = round((coords_Y[int(size/2)] + coords_Y[int(size/2 - 1)]) / 2)
        return mid_x, mid_y
    else:
        mid_x = coords_X[int(size/2)]
        mid_y = coords_Y[int(size/2)]
        return mid_x, mid_y


def gradient(pt1, pt2) -> float:
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def angle(pt1, pt2, pt3) -> float:
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    ang_rads = math.atan(abs((m1-m2)/(1 + m1*m2)))
    return math.degrees(ang_rads)


def get_curvature_and_positional_data() -> (pd.DataFrame, pd.DataFrame):
    curvature_d, positional_d = [], []
    for exp_date in os.listdir(directory):
        if exp_date != "SoftAct11-06-21":
            continue
        exp_date_path = os.path.join(directory, exp_date)
        for exp_num in os.listdir(exp_date_path):
            exp_path = os.path.join(exp_date_path, exp_num)
            for image_name in os.listdir(exp_path):
                img_path = os.path.join(exp_path, image_name)

                # load the image
                img = cv2.imread(img_path)

                # crop the image if it's from the second set
                if exp_date == "SoftAct11-06-21":
                    margin = 200
                    img = img[margin:-margin, margin:-margin]

                # apply contrast
                alpha = 1.1  # Contrast control (1.0-3.0)
                beta = -30  # Brightness control (0-100)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

                # image size = 1/2 * original size
                height, width, _ = img.shape
                new_height, new_width = int(height / 2), int(width / 2)
                img = cv2.resize(img, (new_width, new_height))

                # set lower and upper range for the green marker
                lower_range = np.array([70, 50, 90])
                upper_range = np.array([100, 255, 240])

                # get only the marker areas from the image using a color threshold
                mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_range, upper_range)

                # dilate mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

                # find contours in the thresholded image
                cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = list(filter(lambda cnt: cv2.contourArea(cnt) > 50, cnts))
                cnts = sort_contours(cnts, sort_by_distance=True, sort_by_top_left=True)

                # get the middle points of all marker areas
                blank_img = np.zeros((new_height, new_width, 3), np.uint8)
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
                    cv2.circle(blank_img, (cX, cY), 1, (255, 255, 255), -1)
                    cv2.putText(blank_img, f"tile {i}: {(cX, cY)}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
                    i += 1

                # Get the bending angle in degrees
                df = pd.DataFrame(d)

                # Try to fit circle to points
                all_points = df.drop('Chamber', axis=1).values[1:]
                fitted_circle = circle_fit_by_taubin(all_points)

                # Draw the fitted circle on the image
                cv2.circle(blank_img, (int(fitted_circle[0][0]), int(fitted_circle[0][1])), int(fitted_circle[1]), (255,255, 255), 1)

                coords_X, coords_Y = df['X-Value'].tolist(), df['Y-Value'].tolist()
                print(image_name, coords_X, coords_Y)

                curvatures = []
                for i in range(0, 12, 3):
                    a = (coords_X[i], coords_Y[i])
                    b = (coords_X[i + 1], coords_Y[i + 1])
                    c = (coords_X[i + 2], coords_Y[i + 2])
                    curvatures.append(menger_curvature(a, b, c))

                curvature_d.append({
                    "image": image_name,
                    **{f"curvature {i+1}": curvatures[i] for i in range(len(curvatures))}
                })

                positional_d.append({
                    "image": image_name,
                    ** {f"chamber {i+1} X": coords_X[i] - coords_X[0] for i in range(2, 12)},
                    ** {f"chamber {i+1} Y": coords_Y[i] - coords_Y[0] for i in range(2, 12)}
                })

                # # Debugging:
                # cv2.imshow('Image', img)                                      # TODO: Remove this later
                # cv2.imshow('Mask', mask)                                      # TODO: Remove this later
                # cv2.imshow(f"{image_name} - dots", blank_img)                 # TODO: Remove this later
                # cv2.waitKey(0)
    return pd.DataFrame(curvature_d), pd.DataFrame(positional_d)


if __name__ == '__main__':
    curvature_data, positional_data = get_curvature_and_positional_data()
    print(f"Average distance between {avg_dist_n} chambers: {avg_dist}")
    print(curvature_data)
    print(positional_data)
    cv2.destroyAllWindows()
