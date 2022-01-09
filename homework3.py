import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

src_images = ['src_0.jpg', 'src_1.jpg', 'src_2.jpg']
dst_images = ['dst_0.jpg', 'dst_1.jpg']

# find the distance between two keypoints 
def distance_kps(kp1, kp2):
    return np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))

for src_image in src_images:
    for dst_image in dst_images:

        img1 = cv.imread('HW3_Data/'+src_image) # queryImage
        img2 = cv.imread('HW3_Data/'+dst_image) # trainImage

        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Draw the keypoints on both the images
        resimage1 = cv.drawKeypoints(img1, kp1, 0, (255,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        resimage2 = cv.drawKeypoints(img2, kp2, 0, (255,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the image with drawn keypoints 
        cv.imwrite("HW3_Output/SIFT_kps"+src_image, resimage1)
        cv.imwrite("HW3_Output/SIFT_kps"+dst_image, resimage2)

        # count number of keypoints in both the images.
        no_of_kpts1 = len(kp1)
        no_of_kpts2 = len(kp2)

        # print the number of keypoints found in both images
        print("No of Key Points found using SIFT in image ",src_image,no_of_kpts1)
        print("No of Key Points found using SIFT in image ",dst_image,no_of_kpts2)
        
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # Sort the good matches list
        good = sorted(good, key=lambda f: f[0].distance)
        draw_params = dict(matchColor=(0,255,255), singlePointColor=None, outImg=np.array([]), flags=2)

        # Counting the number of good matches  
        print("Number of good matches found for "+ src_image+ " "+ dst_image, len(good))

        # Drawing only the top 20 matches on the image
        top_20_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, 
                            good[:20], **draw_params)
        cv.imwrite("HW3_Output/Top_20"+ src_image[:-4]+"_"+dst_image, top_20_matches)
        

        # if there are a sufficient number good matches apply homography operation with RANSAC
        if len(good)>10:
            src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)                   # returns homography matrix and mask
            h,w,d = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)          # get boundaries of the image 
            dst = cv.perspectiveTransform(pts,M)
            img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
       
        count = np.sum(mask)

        # using the mask obtained from findHomography 
        print("No of inliers found for image combination", src_image, " ", dst_image, count)
        def dist_kp(kp1, kp2):
            return np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))

        #img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_params)
        #plt.imshow(img3, 'gray'),plt.show()

        # transform and plot the bounding box of the query image found in train image 
        kp_warp = cv.perspectiveTransform(np.array([[[kp.pt[0], kp.pt[1]] for kp in kp1]]), M)[0]
        kp_warp = np.array([cv.KeyPoint(pt[0], pt[1], 1) for pt in kp_warp.tolist()])

        # apply the homography transformation on all kpoints matched and sort it in order of min error.
        matches.sort(key = lambda match: distance_kps(kp_warp[match[0].queryIdx], kp2[match[0].trainIdx]))


        # calculate source key points after homography transformation
        kp_proj_src = cv.perspectiveTransform(np.array([[[kp.pt[0], kp.pt[1]] for kp in kp1]]),M)[0]
        kp_proj_src = np.array([cv.KeyPoint(pt[0], pt[1], 1) for pt in kp_proj_src.tolist()])

        matches_after_homo = sorted(good, key = lambda match: dist_kp(kp_proj_src[match[0].queryIdx], kp2[match[0].trainIdx]))
        draw_params = dict(matchColor = (0,255,255), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = None, # draw only inliers
                        flags = 2)
        # show top 10 matches after homography transformation
        img_matches_after = cv.drawMatchesKnn(img1, kp1,
                        img2, kp2,
                        matches_after_homo[:10], None, **draw_params)
                
        cv.imwrite("HW3_Output/Top_10RANSAC"+ src_image[:-4]+"_"+dst_image, img_matches_after)
        print("Homography Matrix for "+ src_image+" "+ dst_image)
        print(M)
        print("######################")
