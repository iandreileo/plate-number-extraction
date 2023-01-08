import numpy as np
import cv2
import pytesseract
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
from itertools import product


class Canny:

    def grayscale(self,image):

        # Transformam toti pixelii in pixeli gray
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray


    def gen_gaussian_kernel(self, k_size, sigma):
        # Calculam kernelul gaussian conform teoriei
        # Sursa(1): https://pages.stat.wisc.edu/~mchung/teaching/MIA/reading/diffusion.gaussian.kernel.pdf
        center = k_size // 2
        x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
        return g


    def gaussian_filter(self, image, k_size, sigma):

        # Extragem inaltime si latimea imaginii
        height, width = image.shape[0], image.shape[1]
        
        # Calculam latimea si inaltime
        # Luand in considerare parametrul de nucleu furnizat
        dst_height = height - k_size + 1
        dst_width = width - k_size + 1

        # Construim imaginea conform teoriei
        image_array = zeros((dst_height * dst_width, k_size * k_size))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = ravel(image[i : i + k_size, j : j + k_size])
            image_array[row, :] = window
            row += 1

        # Transformam kernelul intr-o forma (k patrat, 1)
        gaussian_kernel = self.gen_gaussian_kernel(k_size, sigma)
        filter_array = ravel(gaussian_kernel)

        # Facem produsul celor 2: imagine si filtrul
        # Facem reshape
        # Transformam rezultatul in 8 canale
        dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

        # Returnam
        return dst

    def blur(self, image):
        image = self.gaussian_filter(image, 5, sigma=0.8)

        return image

    def sobel(self, image):
        # Aplicam pe imagine grayscale
        # Apoi ii aplicam un gaussian blur
        # Aceste 2 operatii pentru netezirea imaginii
        image = self.blur(self.grayscale(image))

        # Initializm o matrice de convolutie
        # de marimea imaginii, dar cu pixelii setati pe 0
        convolved = np.zeros(image.shape)

        # Initializam 2 matrici
        # Una petru orizontal, una pentru vertical
        # Sursa(1): https://en.wikipedia.org/wiki/Sobel_operator
        # Sursa(2): Lab SPG
        G_x = np.zeros(image.shape)
        G_y = np.zeros(image.shape)
        
        # Extragem tuplul cu marimea imaginii
        size = image.shape

        # Declaram la fel ca mai sus
        # 2 matrici orizontal, si vertical pentru kernenul filtrului 
        # Sursa(1): Lab SPG
        # Sursa(2): Implementare matlab https://en.wikipedia.org/wiki/Sobel_operator
        kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
        kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))


        # Completam matricile orizontale si verticale
        # Conform teoriei
        # https://en.wikipedia.org/wiki/Sobel_operator
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                G_x[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_x))
                G_y[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_y))
        
        # Pentru a forma imaginea la loc
        # Vom folosi operatia de convolutie conform teoriei
        convolved = np.sqrt(np.square(G_x) + np.square(G_y))
        convolved = np.multiply(convolved, 255.0 / convolved.max())

        angles = np.rad2deg(np.arctan2(G_y, G_x))
        angles[angles < 0] += 180

        # Transformam imaginea in 8 canale
        convolved = convolved.astype('uint8')

        # Returnam imaginea si unghiurile
        return convolved, angles

    def nms(self, image, angles):
        # Extragem tuplul cu dimensiunile
        size = image.shape

        # Initializam o matrice de
        # dimensiunea imaginii
        suppressed = np.zeros(size)

        # Iteram prin pixelii imaginii
        # Si incercam sa gasim 
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    value_to_compare = max(image[i, j - 1], image[i, j + 1])
                elif (22.5 <= angles[i, j] < 67.5):
                    value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
                elif (67.5 <= angles[i, j] < 112.5):
                    value_to_compare = max(image[i - 1, j], image[i + 1, j])
                else:
                    value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
                
                if image[i, j] >= value_to_compare:
                    suppressed[i, j] = image[i, j]

        suppressed = np.multiply(suppressed, 255.0 / suppressed.max())


        return suppressed

    def double_threshold_hysteresis(self, image, low, high):

        # Definim tresholdul minim si maxim
        weak = 50
        strong = 255

        # Extragm tuplul cu dimensiunile imaginii
        size = image.shape

        # Initializam rezultatul ca fiind o matrice
        # De dimensiunea imaginii initiale
        result = np.zeros(size)

        # Extragem edge-urile care se incadreaza la weak si storng
        weak_x, weak_y = np.where((image > low) & (image <= high))
        strong_x, strong_y = np.where(image >= high)

        # Setam in matricea rezultat
        # Calculele anterioare
        result[strong_x, strong_y] = strong
        result[weak_x, weak_y] = weak

        # Definim directiile orizontale si verticale
        dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
        dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
        size = image.shape
        
        # Calculam edge-urile care raman in imagine
        while len(strong_x):
            x = strong_x[0]
            y = strong_y[0]
            strong_x = np.delete(strong_x, 0)
            strong_y = np.delete(strong_y, 0)
            for direction in range(len(dx)):
                new_x = x + dx[direction]
                new_y = y + dy[direction]
                if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                    result[new_x, new_y] = strong
                    np.append(strong_x, new_x)
                    np.append(strong_y, new_y)
        
        result[result != strong] = 0
        return result

    def canny(self, image, low, high):

        # Pentru a aplica filtrul CANNY
        # Trebuie sa urmam pasii de mai jos:

        # Aplicam sobel pe imagine
        image, angles = self.sobel(image)
        
        # Aplicam NMS (non-maxima supression)
        # Pentru a minimiza contururile ne-necesare
        image = self.nms(image, angles)
                
        # Aplicam Hysteresis thresholding
        # Pentru a netezi imaginea si a scoate in evidenta contururile
        image = self.double_threshold_hysteresis(image, low, high)
        
        return image

    def ocr(self, ROI):

        nr_inmatriculare = pytesseract.image_to_string(ROI)
        return nr_inmatriculare

    def find_contours(self, image, new_image, show=True):
        imagecontours, _ = cv2.findContours(new_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imagecontours = sorted(imagecontours, key=cv2.contourArea, reverse=True)[:20]

        nr_inmatriculare = []
        #for each of the contours detected, the shape of the contours is approximated using approxPolyDP() function and the contours are drawn in the image using drawContours() function
        for count in imagecontours:
            epsilon = 0.01 * cv2.arcLength(count, True)
            approximations = cv2.approxPolyDP(count, epsilon, True)
            #the name of the detected shapes are written on the image
            i, j = approximations[0][0] 
            if len(approximations) == 4:
                x,y,w,h = cv2.boundingRect(count)
                ROI = image[y:y+h, x:x+w]

                # TODO: De pus o conditie de ratia rectangulara
                if show:
                    cv2.imshow('INMATRICULARE',ROI)
                    cv2.waitKey()

                # extragem folosind OCR
                current_plate_number = self.ocr(ROI)
                nr_inmatriculare.append(current_plate_number)

                if show:
                    cv2.drawContours(image, [approximations], 0, (0,255,0), 2)
                    cv2.putText(image, "NR. INMATRICULARE: " + current_plate_number, (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        #displaying the resulting image as the output on the screen
        if show:
            cv2.imshow("Resulting_image", image)
            cv2.waitKey(0)

        return nr_inmatriculare


    def extract_plate_number(self, image, show=True):

        # Generam imaginea folosind canny
        new_image = self.canny(image, 0, 50)

        # Setam sa fie pe 8 canale
        # Ca sa arate bine pe imshow
        new_image = new_image.astype(np.uint8)

        if show:
            cv2.imshow("Sobel Image", new_image)
            cv2.waitKey(0)

        # Returnam si afisam numarul de inamtriculare
        return self.find_contours(image, new_image, show)