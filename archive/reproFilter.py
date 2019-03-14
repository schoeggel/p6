import cv2

# original code in c++: P4 Trainscanner FeatureFilter.cpp (C) Brütsch Tobias / Burri Marcel

def reprojectionErrorFromLines(line1, line2, point1, point2):
    a2 = line2[0]   # epipolar line
    b2 = line2[1]
    c2 = line2[2]
    norm_factor2 = (a2**2 + b2**2)**0.5

    a1 = line1[0]
    b1 = line1[1]
    c1 = line1[2]
    norm_factor1 = (a1**2 + b1**2)**0.5

    return abs(point1[0] * a2 + point1[1] * b2 + c2) / norm_factor2 + abs(point2[0] * a1 + point2[1] * b1 + c1) / norm_factor1


def filterReprojectionError(matches, F, pts1, pts2, maxDistance = 5):
    lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = lines2.reshape(-1, 3)
    goodMatches = []

    for i, match in enumerate(matches):
        err = abs(reprojectionErrorFromLines(lines1[i], lines2[i], pts1[i], pts2[i]))
        if err < maxDistance:
            goodMatches.append(match)

    filterinfo = "reproFilter before/after (maxDistance=" + str(maxDistance) + ": "
    filterinfo += str(matches.__len__()) + " / " + str(goodMatches.__len__())
    return goodMatches, filterinfo