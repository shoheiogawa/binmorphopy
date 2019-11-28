#!/usr

'''
Test the fast binary morphological operation functions
'''

import unittest

import numpy as np
import skimage.morphology

from binmorphopy.morphology import binary_dilation, binary_erosion, \
        binary_closing, binary_opening, _get_perimeter_image

class TestMorphology(unittest.TestCase):

    def _compare(self, image, funcs, selem_args=None, out_args=None):
        if not len(funcs) == len(selem_args) == len(out_args):
            raise RuntimeError('The number of items in the arugments do not match.')
        results = []
        for func, selem, out in zip(funcs, selem_args, out_args):
            out_result = func(image, selem=selem, out=out)
            if out is not None:
                out_result = out
            results.append(out_result.copy())
        for i in range(len(results) - 1):
            if np.all(results[i] == results[i + 1]):
                pass
            else:
                return False
        return True

    def test_binary_dilation(self):
        '''Test the binary-dilation function using random and structured images
        '''
        # 1D with a structuring element of radius at 1
        structure = np.random.randint(0, 2, size=[10]) > 0
        selem = np.ones(shape=[3])
        test_data = binary_dilation(structure, selem)
        skiamge_data = skimage.morphology.binary_dilation(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 1D with a structuring element of radius at 1
        structure = np.random.randint(0, 2, size=[10]) > 0
        selem = np.ones(shape=[3])
        funcs = [binary_dilation, skimage.morphology.binary_dilation]
        selem_args = [selem, selem]
        out_args = [None, None]
        self.assertTrue(self._compare(structure, funcs, selem_args, out_args))

        # 2D with a structuring element of radius at 1
        structure = np.random.randint(0, 2, size=[10, 10]) > 0
        selem = skimage.morphology.disk(1)
        test_data = binary_dilation(structure, selem)
        skiamge_data = skimage.morphology.binary_dilation(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a structuring element of radius at 2
        structure = np.zeros(shape=[10, 10])
        structure[3:5, 3:5] = 1
        selem = skimage.morphology.disk(2)
        test_data = binary_dilation(structure, selem)
        skiamge_data = skimage.morphology.binary_dilation(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a structuring element of square
        structure = np.zeros(shape=[14, 14])
        structure[5:7, 5:7] = 1
        selem = np.ones(shape=[5, 5])
        test_data = binary_dilation(structure, selem)
        skiamge_data = skimage.morphology.binary_dilation(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with the default structuring element
        structure = np.random.randint(0, 2, size=[10, 10]) > 0
        test_data = binary_dilation(structure)
        skiamge_data = skimage.morphology.binary_dilation(structure)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a default structuring element with an output array argument
        structure = np.random.randint(0, 2, size=[10, 10]) > 0
        test_data = np.zeros([10, 10], dtype=np.bool)
        binary_dilation(structure, out=test_data)
        skiamge_data = skimage.morphology.binary_dilation(structure)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 3D with a structuring element of radius at 1
        structure = np.random.randint(0, 2, size=[10, 10, 10]) > 0
        selem = skimage.morphology.ball(1)
        test_data = binary_dilation(structure, selem)
        skiamge_data = skimage.morphology.binary_dilation(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 3D with a structuring element of radius at 2
        structure = np.random.randint(0, 2, size=[10, 10, 10]) > 0
        selem = skimage.morphology.ball(2)
        test_data = binary_dilation(structure, selem)
        skiamge_data = skimage.morphology.binary_dilation(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 3D with the default structuring element
        structure = np.random.randint(0, 2, size=[10, 10, 10]) > 0
        test_data = binary_dilation(structure)
        skiamge_data = skimage.morphology.binary_dilation(structure)
        self.assertEqual((test_data == skiamge_data).all(), True)

    def test_binary_erosion(self):
        '''Test the binary-erosion function using random and structured images
        '''
        # 1D with a structuring element of radius at 1
        structure = np.random.randint(0, 3, size=[10, 10]) > 0
        selem = skimage.morphology.disk(1)
        test_data = binary_erosion(structure, selem)
        skiamge_data = skimage.morphology.binary_erosion(structure, selem)
        self.assertEqual((test_data== skiamge_data).all(), True)

        # 2D with a structuring element of radius at 2
        structure = np.zeros(shape=[10, 10])
        structure[3:5, 3:5] = 1
        selem = skimage.morphology.disk(2)
        test_data = binary_erosion(structure, selem)
        skiamge_data = skimage.morphology.binary_erosion(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a default structuring element
        structure = np.random.randint(0, 3, size=[10, 10]) > 0
        test_data = binary_erosion(structure)
        skiamge_data = skimage.morphology.binary_erosion(structure)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a default structuring element with an output array argument
        structure = np.random.randint(0, 2, size=[10, 10]) > 0
        test_data = np.zeros([10, 10], dtype=np.bool)
        binary_erosion(structure, out=test_data)
        skiamge_data = skimage.morphology.binary_erosion(structure)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 3D with the default structuring element
        structure = np.random.randint(0, 2, size=[10, 10, 10]) > 0
        test_data = binary_erosion(structure)
        skiamge_data = skimage.morphology.binary_erosion(structure)
        self.assertEqual((test_data == skiamge_data).all(), True)

    def test_binary_closing(self):
        '''Test the binary-closing function using random and structured images
        '''
        # 1D with a structuring element of radius at 1
        structure = np.random.randint(0, 3, size=[10, 10]) > 0
        selem = skimage.morphology.disk(1)
        test_data = binary_closing(structure, selem)
        skiamge_data = skimage.morphology.binary_closing(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a structuring element of radius at 2
        structure = np.zeros(shape=[20, 20])
        structure[4:8, 4:8] = 1
        structure[9:12, 9:12] = 1
        selem = skimage.morphology.disk(2)
        test_data = binary_closing(structure, selem)
        skiamge_data = skimage.morphology.binary_closing(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a default structuring element with an output array argument
        structure = np.random.randint(0, 2, size=[10, 10]) > 0
        test_data = np.zeros([10, 10], dtype=np.bool)
        binary_closing(structure, out=test_data)
        skiamge_data = skimage.morphology.binary_closing(structure)
        self.assertEqual((test_data == skiamge_data).all(), True)

    def test_binary_opening(self):
        '''Test the binary-opening function using random and structured images
        '''
        # 1D with a structuring element of radius at 1
        structure = np.random.randint(0, 3, size=[10, 10]) > 0
        selem = skimage.morphology.disk(1)
        test_data = binary_opening(structure, selem)
        skiamge_data = skimage.morphology.binary_opening(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a structuring element of radius at 2
        structure = np.zeros(shape=[20, 20])
        structure[4:8, 4:8] = 1
        structure[8:12, 8:12] = 1
        selem = skimage.morphology.disk(2)
        test_data = binary_opening(structure, selem)
        skiamge_data = skimage.morphology.binary_opening(structure, selem)
        self.assertEqual((test_data == skiamge_data).all(), True)

        # 2D with a default structuring element with an output array argument
        structure = np.random.randint(0, 2, size=[10, 10]) > 0
        test_data = np.zeros([10, 10], dtype=np.bool)
        binary_opening(structure, out=test_data)
        skiamge_data = skimage.morphology.binary_opening(structure)
        self.assertEqual((test_data == skiamge_data).all(), True)

    def test_get_perimeter_image(self):
        '''Test the function that extracts the perimeter structure of the input image
        '''
        structure = np.zeros(shape=[10, 10], dtype=np.bool)
        perimeter = _get_perimeter_image(structure)
        self.assertEqual((structure == perimeter).all(), True)

        structure = np.ones(shape=[10, 10], dtype=np.bool)
        perimeter = _get_perimeter_image(structure)
        self.assertEqual(perimeter.sum(), 0)

        structure = np.zeros(shape=[10, 10], dtype=np.bool)
        structure[4:6, 4:6] = True
        perimeter = _get_perimeter_image(structure)
        self.assertEqual((structure == perimeter).all(), True)
        self.assertEqual(perimeter.sum(), 4)

        structure = np.zeros(shape=[10, 10], dtype=np.bool)
        structure[4:7, 4:7] = True
        perimeter = _get_perimeter_image(structure)
        self.assertEqual(perimeter[5, 5], False)
        self.assertEqual(perimeter.sum(), 8)

        structure = np.zeros(shape=[10, 10], dtype=np.bool)
        structure[4:7, :] = True
        perimeter = _get_perimeter_image(structure)
        self.assertEqual(perimeter.sum(), 20)
        self.assertEqual((perimeter[5, :] == False).all(), True)

if __name__ == '__main__':
    unittest.main()
