# Wie verhält sich opencv beim TemplateMatching bezüglich den Farbkanälen?
# Hintergrund ist die Idee, via individueller Kanalbefüllung einen Masken-
# effekt zu erhalten.So dass der ungültige Suchbereich im Bild wie auch der
# zu ignorierende Teil des Templates wie gewünscht funktionieren.


import cv2
import numpy as np
from matplotlib import pyplot as plt

#Test1
imagenames = ['01', '02', '03', '04']
templatenames = ['11', '12', '13', '14', '15']

#Test 2
imagenames = ['21', '22', '23', '24','25','26','27']
templatenames = ['31','32']


#Test 3
imagenames = ['41', '42', '43', '44','45','46','47']
templatenames = ['32']

combos = [(x,y) for x in templatenames for y in imagenames]
for c in combos:
    tname, iname = c
    print(f'Loop: {tname} - {iname}')

    img = cv2.imread(f'data/tmtest/{iname}.png')
    template = cv2.imread(f'data/tmtest/{tname}.png')
    img2 = img.copy()
    h, w = template.shape[0:2]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #methods = ['cv2.TM_CCOEFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            val=min_val
        else:
            top_left = max_loc
            val=max_val

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)


        plt.subplot(311),plt.imshow(template,cmap = 'gray')
        plt.title('Template'), plt.xticks([]), plt.yticks([])
        plt.subplot(312),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(313),plt.imshow(img,cmap = 'gray')
        plt.title(f'Detected Point (score: {val})'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        fig = plt.gcf()
        fig.set_size_inches(8,10)
        plt.savefig(f'tmp/plt-{iname}-{tname}-{method}.png', dpi=150)
        #plt.show()