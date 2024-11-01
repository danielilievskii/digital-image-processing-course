import glob
import os
import cv2
import numpy as np

def load_images(directory: str) -> list:
    image_files = glob.glob(os.path.join(directory, '*'))
    return [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]

def compute_sift_descriptors(image: np.ndarray):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)

def find_good_matches(desc1, desc2, ratio=0.75) -> list:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in matches if m.distance < ratio * n.distance]

def resize_images_to_same_height(img1, img2):
    height1, width1 = img1.shape
    height2, width2 = img2.shape

    if height1 > height2:
        img2 = cv2.resize(img2, (width2 * height1 // height2, height1))
    else:
        img1 = cv2.resize(img1, (width1 * height2 // height1, height2))

    return img1, img2

def resize_image_to_fit_screen(image, max_height=800):
    height, width = image.shape[:2]
    if height > max_height:
        new_width = int(width * max_height / height)
        image = cv2.resize(image, (new_width, max_height))
    return image

def draw_results(query_img, query_kp, best_img, best_kp, matches):
    query_img, best_img = resize_images_to_same_height(query_img, best_img)

    original_images_concat = np.concatenate((query_img, best_img), axis=1)

    query_img_color = cv2.cvtColor(query_img, cv2.COLOR_GRAY2RGB)
    best_img_color = cv2.cvtColor(best_img, cv2.COLOR_GRAY2RGB)

    best_keypoints_img = cv2.drawKeypoints(best_img_color, best_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    query_keypoints_img = cv2.drawKeypoints(query_img_color, query_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    keypoints_images_concat = np.concatenate((query_keypoints_img, best_keypoints_img), axis=1)

    matches_images = cv2.drawMatches(query_img_color, query_kp, best_img_color, best_kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    original_images_concat = resize_image_to_fit_screen(original_images_concat)
    keypoints_images_concat = resize_image_to_fit_screen(keypoints_images_concat)
    matches_images = resize_image_to_fit_screen(matches_images)

    cv2.imshow('Original', original_images_concat)
    cv2.imshow('Keypoints', keypoints_images_concat)
    cv2.imshow('Matches', matches_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    poster_images = load_images('Database2')
    poster_descriptors = [compute_sift_descriptors(image) for image in poster_images]

    query_image_name = input('Vnesete go imeto na slikata (primer: hw7_poster_1.jpg) : ')
    query_image = cv2.imread(query_image_name, cv2.IMREAD_GRAYSCALE)
    query_kp, query_desc = compute_sift_descriptors(query_image)

    best_matches = []
    best_database_index = -1
    max_good_matches = 0

    for i, (database_keypoints, database_descriptors) in enumerate(poster_descriptors):
        good_matches = find_good_matches(query_desc, database_descriptors)
        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_matches = good_matches
            best_database_index = i

    best_database_image = poster_images[best_database_index]
    best_database_keypoints = poster_descriptors[best_database_index][0]

    draw_results(query_image, query_kp, best_database_image, best_database_keypoints, best_matches)


if __name__ == '__main__':
    main()
