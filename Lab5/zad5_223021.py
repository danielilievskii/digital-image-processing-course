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

def draw_results(query_img, query_kp, best_img, best_kp, matches, matches_mask):
    query_keypoints_img = cv2.drawKeypoints(query_img, query_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    best_keypoints_img = cv2.drawKeypoints(best_img, best_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    matches_img = cv2.drawMatches(query_img, query_kp, best_img, best_kp, matches, None, matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Query Image Keypoints', query_keypoints_img)
    cv2.imshow('Best Match Image Keypoints', best_keypoints_img)
    cv2.imshow('Matches', matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    poster_images = load_images('Database2')
    poster_descriptors = [compute_sift_descriptors(image) for image in poster_images]

    query_image_name = input('Vnesete go imeto na slikata (primer: hw7_poster_1.jpg) : ')
    query_image = cv2.imread(query_image_name, cv2.IMREAD_GRAYSCALE)
    query_kp, query_desc = compute_sift_descriptors(query_image)

    matches_per_poster = [find_good_matches(query_desc, desc[1]) for desc in poster_descriptors]
    best_match_index = np.argmax([len(matches) for matches in matches_per_poster])

    best_matches = matches_per_poster[best_match_index]
    poster_kp = poster_descriptors[best_match_index][0]
    src_pts = np.float32([query_kp[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([poster_kp[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    draw_results(query_image, query_kp, poster_images[best_match_index], poster_kp, best_matches, matches_mask)


if __name__ == '__main__':
    main()