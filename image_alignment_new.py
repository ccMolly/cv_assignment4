import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def ransac(T):
    trans_param = None
    maxInliers = []
    count = 0
    while count < 100:
        try:
            p = []
            A = []
            b = []
            inliers = []
            for j in range(3):
                p.append(T[random.randrange(0, len(T))])
            for j in range(3):
                p1 = [p[j][0], p[j][1]]
                p2 = [p[j][2], p[j][3]]
                A.append([p1[0], p1[1], 1, 0, 0, 0])
                A.append([0, 0, 0, p1[0], p1[1], 1])
                b.append(p2[0])
                b.append(p2[1])
            A = np.array(A)
            #print(A)
            b = np.array(b)
            #print(b)
            x = np.linalg.solve(A, b)
            #print(x)
            x1 = x.tolist()
            x1.append(0)
            x1.append(0)
            x1.append(1)
            x1 = np.array(x1)
            h = np.reshape(x1, (3, 3))
            for j in range(len(T)):
                c = np.transpose(np.matrix([T[j][0], T[j][1], 1]))
                gt = np.transpose(np.matrix([T[j][2], T[j][3], 1]))
                est = np.dot(h, c)
                est = (1 / est.item(2)) * est
                loss = np.linalg.norm(est - gt)
                if loss < 10:
                    inliers.append(T[j])
            if len(inliers) > len(maxInliers):
                maxInliers = inliers
                print(len(maxInliers))
                trans_param = np.reshape(x, (2, 3))
            count += 1
        except:
            continue

    for i in range(int(len(maxInliers)/3)):
        A = []
        b = []
        inliers = []
        try:
            for j in range(3):
                p = maxInliers[3*i+j]
                p1 = [p[0], p[1]]
                p2 = [p[2], p[3]]
                A.append([p1[0], p1[1], 1, 0, 0, 0])
                A.append([0, 0, 0, p1[0], p1[1], 1])
                b.append(p2[0])
                b.append(p2[1])
            A = np.array(A)
            b = np.array(b)
            x = np.linalg.solve(A, b)
            x1 = x.tolist()
            x1.append(0)
            x1.append(0)
            x1.append(1)
            x1 = np.array(x1)
            h = np.reshape(x1, (3, 3))
            for j in range(len(T)):
                c = np.transpose(np.matrix([T[j][0], T[j][1], 1]))
                gt = np.transpose(np.matrix([T[j][2], T[j][3], 1]))
                est = np.dot(h, c)
                est = (1 / est.item(2)) * est
                #print(est)
                loss = np.linalg.norm(est - gt)
                if loss < 10:
                    inliers.append(T[j])
            if len(inliers) > len(maxInliers):
                maxInliers = inliers
                trans_param = np.reshape(x, (2, 3))
        except:
            continue
    return maxInliers, trans_param


def drawMatches(img1, img2, inliers):
    row1, col1 = img1.shape[:2]
    #print(row1, col1)
    row2, col2 = img2.shape[:2]
    #print(row2, col2)
    output = np.zeros((max(row1,row2), col1+col2,3),dtype='uint8')
    output[:row1, :col1, :] = img1
    output[:row2, col1:col1+col2, :] = img2
    for inlier in inliers:
        #print(inlier[0])
        #print(int(inlier[0]))
        cv2.line(output,(int(inlier[0]), int(inlier[1])), (int(inlier[2]+col1),int(inlier[3])),(0,0,255),1)
    return output


sift = cv2.xfeatures2d.SIFT_create()
img_1 = cv2.imread("scene.pgm")
img_2 = cv2.imread("book.pgm")
kp_1, des_1 = sift.detectAndCompute(img_1, None)
kp_2, des_2 = sift.detectAndCompute(img_2, None)

img_1 = cv2.drawKeypoints(img_1, kp_1, img_1, color=(255, 0, 255))
img_2 = cv2.drawKeypoints(img_2, kp_2, img_2, color=(255, 0, 255))

# cv2.imwrite('img1_keypoints.jpg',img_1)
# cv2.imwrite('img2_keypoints.jpg',img_2)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des_1, des_2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.9 * n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img_1, kp_1, img_2, kp_2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
pairs = []
for m in good:
    pairs.append([kp_1[m[0].queryIdx].pt[0], kp_1[m[0].queryIdx].pt[1], kp_2[m[0].trainIdx].pt[0], kp_2[m[0].trainIdx].pt[1]])

inlier_set, trans = ransac(pairs)
print(trans)

#M = np.matrix([[trans.item(0),trans.item(1),trans.item(4)],[trans.item(2),trans.item(3),trans.item(5)]])
out = drawMatches(img_1, img_2, inlier_set)
plt.imshow(out),plt.show()
transfered = cv2.warpAffine(img_1, trans, (1024,1024), borderValue=(255, 255, 255))
plt.imshow(transfered),plt.show()

