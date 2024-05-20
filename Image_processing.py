import cv2
import numpy as np

class Image_processing:
    def __init__(self) -> None:
        '''
        Constructor of image processing class. Doesn't require any arguments.
        '''
        self.size : int = 100
        self.vid = cv2.VideoCapture(0)

        self.capture_name  : str = "Video capture"
        self.Trackbar_name : str = "Trackbar"

        _, self.capture = self.vid.read()
        self.capture = self.resize(self.size)

        self.grey_scale   : bool = False
        self.canny_filter : bool = False
        self.sobel_filter : int  = 0

    def rename_capture(self, name: str) -> None:
        '''
        Rename window of captured video.
        
        Args:
            name (str): Name of the window.'''
        self.capture_name = name

    def rename_trackbar(self, name: str) -> None:
        '''
        Rename window of trackbars.
        
        Args:
            name (str): Name of the window.'''
        self.Trackbar_name = name

    def window_config(self) -> None:
        '''
        Configuration function for windows. It creates trackbars. Doesn't require any values.'''
        cv2.destroyAllWindows()

        cv2.namedWindow(self.capture_name)
        cv2.namedWindow(self.Trackbar_name)
        
        cv2.createTrackbar('Red intensity'  , self.Trackbar_name, 0, 150, lambda x: print(f"Red intesity set to: {x}%"))
        cv2.createTrackbar('Green intensity', self.Trackbar_name, 0, 150, lambda x: print(f"Green intesity set to: {x}%"))
        cv2.createTrackbar('Blue intensity' , self.Trackbar_name, 0, 150, lambda x: print(f"Blue intesity set to: {x}%"))

        cv2.createTrackbar('H intensity'    , self.Trackbar_name, 0, 150, lambda x: print(f"H intesity set to: {x}%"))
        cv2.createTrackbar('S intensity'    , self.Trackbar_name, 0, 150, lambda x: print(f"S intesity set to: {x}%"))
        cv2.createTrackbar('V intensity'    , self.Trackbar_name, 0, 150, lambda x: print(f"V intesity set to: {x}%"))

        cv2.createTrackbar('Brightness'     , self.Trackbar_name,   0, 100, lambda x: print(f"Brightness set to: {x}"))
        cv2.createTrackbar('Contrast'       , self.Trackbar_name, 100, 150, lambda x: print(f"Contrast set to: {x}"))
        cv2.setTrackbarMin('Brightness'     , self.Trackbar_name, -100)
        cv2.setTrackbarMin('Contrast'       , self.Trackbar_name, 50)

    def resize(self, size : int, x : int = 4, y : int = 3) -> np.ndarray:
        '''Funciton for resizing window. Default format is 4x3.
        
        Args:
            size (int): Size of the window (will be multiplied by x and y).
            x (int, optional): Width of the window (for aspect ratio).
            y (int, optional): Height of the window (for aspect ratio).
        Returns:
            Numpy array: Resized window.'''
        self.size = size
        return cv2.resize(self.capture,(self.size*x,self.size*y), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)
    
    def cliper(self, x : float, color : int, cap : int = 255) -> None:
        '''
        Method used to calculate color values.
        
        Args:
            x (float): Multiplication factor.
            color (int): Index of the color.
            cap (int, optional): Maximum value for interval (defaults to 255).'''
        self.capture[:,:,color] = np.clip(self.capture[:,:,color] + self.capture[:,:,color]*x ,0, cap)


    def show(self) -> None:
        '''
        Method used to show window. Doesn't require any arguments.'''

        self.window_config()
        while True:
            _, self.capture = self.vid.read()
            self.capture = self.resize(self.size)
            self.capture = np.fliplr(self.capture)

            r : float = cv2.getTrackbarPos('Red intensity'  , self.Trackbar_name)/100
            g : float = cv2.getTrackbarPos('Green intensity', self.Trackbar_name)/100
            b : float = cv2.getTrackbarPos('Blue intensity' , self.Trackbar_name)/100
            
            self.cliper(r, 2)
            self.cliper(g, 1)
            self.cliper(b, 0)

            self.capture = cv2.cvtColor(self.capture, cv2.COLOR_BGR2HSV)

            H : float = cv2.getTrackbarPos('H intensity', self.Trackbar_name)/100
            S : float = cv2.getTrackbarPos('S intensity', self.Trackbar_name)/100
            V : float = cv2.getTrackbarPos('V intensity', self.Trackbar_name)/100
        
            self.cliper(H, 0, cap=179)
            self.cliper(S, 1)
            self.cliper(V, 2)

            self.capture = cv2.cvtColor(self.capture, cv2.COLOR_HSV2BGR)

            Brightness  : int   = cv2.getTrackbarPos('Brightness',self.Trackbar_name)
            Contrast    : float = cv2.getTrackbarPos('Contrast',self.Trackbar_name)/100
            
            self.capture[:,:,0] = np.clip(self.capture[:,:,0]*Contrast + Brightness, 0, 255)
            self.capture[:,:,1] = np.clip(self.capture[:,:,1]*Contrast + Brightness, 0, 255)
            self.capture[:,:,2] = np.clip(self.capture[:,:,2]*Contrast + Brightness, 0, 255)

            key : int = cv2.waitKey(10)

            if 27 == key:
                break

            if 103 == key:
                self.grey_scale = not self.grey_scale
            if 99 == key:
                self.canny_filter = not self.canny_filter
            if 115 == key:
                self.sobel_filter += 1
                self.sobel_filter = self.sobel_filter%3

            if self.grey_scale:
                self.capture = cv2.cvtColor(self.capture, cv2.COLOR_BGR2GRAY)
            if self.sobel_filter > 0:
                if 1 == self.sobel_filter:
                    mask : np.ndarray = np.array([[-1,0,1],
                                                  [-2,0,2],
                                                  [-1,0,1]])

                else:
                    mask : np.ndarray = np.array([[-1,-2,-1],
                                                  [0,0,0],
                                                  [1,2,1]])

                self.capture = cv2.filter2D(self.capture, -1, mask)
            if self.canny_filter:
                self.capture = cv2.Canny(self.capture, 100, 200)
            
            cv2.imshow(self.capture_name, self.capture)

        self.vid.release()
        cv2.destroyAllWindows()

    def  __str__(self):
        self.show()
        return "Image processing class"


def main() -> None:
    image_processing : Image_processing = Image_processing()
    print(image_processing)

if __name__ == '__main__':
    main()