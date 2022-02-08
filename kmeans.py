import cv2
import numpy as np
import argparse


def euc_dist(pixel, centroid):
    return np.linalg.norm(pixel - centroid)


class KMeansModel:

    def __init__(self, img_data, k):
        self.centroids = np.empty(shape=(k, 3), dtype=int)
        self.k = k
        self.width = len(img_data)
        self.height = len(img_data[0])
        self.data = img_data.reshape( (self.width*self.height, 3) )
        for i, coordinate in enumerate(np.random.choice(np.arange(len(self.data)), k)):
            self.centroids[i] = self.data[coordinate]
        self.assignments = np.zeros(len(self.data), dtype=int)

    def assignment(self):
        num_changes = 0
        total_mse = 0
        for i in range(len(self.data)):
            pixel = self.data[i]
            a = 1
            min = euc_dist(pixel, self.centroids[0])
            for j in range(1, self.k):
                dist = euc_dist(pixel, self.centroids[j])
                if dist < min:
                    min = dist
                    a = j+1
            total_mse += min
            if self.assignments[i] != a:
                self.assignments[i] = a
                num_changes += 1
        return num_changes, total_mse

    def update(self):
        for i in range(self.k):
            total = np.zeros(3, dtype=int)
            count = 0
            for j, (pixel, assign) in enumerate(zip(self.data, self.assignments)):
                if assign == i+1:
                    total = total + pixel
                    count += 1
            mean = total / count   
            self.centroids[i] = mean           

    def change_pixels(self):
        for i, (pixel, assign) in enumerate(zip(self.data, self.assignments)):
            self.data[i] = self.centroids[assign-1]

    def unflatten(self):
        self.data = self.data.reshape( (self.width, self.height, 3) )

    def evaluate(self, max_iters): 
        print(f'Training K-means model on image, with k={self.k} clusters/centroids and max_iters={max_iters}  ...')
        for i in range(max_iters):
            (num_changes, total_mse) = self.assignment()
            if num_changes == 0:
                print(f'After step {i+1}: Number of assignment changes: {num_changes}, Total MSE: {total_mse}')
                break
            self.update()
            print(f'After step {i+1}: Number of assignment changes: {num_changes}, Total MSE: {total_mse}')
        
        self.change_pixels()
        self.unflatten()
        

def main():
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--file', type=str, default='fruits.jpg')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--max_iters', type=int, default=35)
    args = parser.parse_args()
    
    img = cv2.imread(args.file)
    kmeans = KMeansModel(img, k=args.k)
    kmeans.evaluate(max_iters=args.max_iters)

    cv2.imwrite(f'kmeans_{args.file}_k-{args.k}_iters-{args.max_iters}.jpg', kmeans.data)
    cv2.imshow("kmeans", kmeans.data)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()


    