# -*- coding: utf-8 -*-
"""
A library for doing arithmetic with versors (unit quaternion), which can be
used to represent 3D rotations or orientation in 3D. Provides a Versor class
class for these purposes.

@author: Will D. Spann <willdspann@gmail.com>
 
@version 0.1
         
Copyright 2015 W. Spann Systems Consulting
"""

import math
import numpy as np
import numpy.linalg as la
import numpy.matlib as ml


# Global Constants:

DEF_THRESHOLD = 1.0e-10  # Default threshold for equality comparisons
NUMERIC_TYPE = (int, float, np.float32, np.float64)


class Versor(object):
    """
    This class represents a unit quaternion (a.k.a. a versor), in the JPL
    notation. This notation represents versors as a 4-vector, with the imaginary
    components coming first and the real component listed last, as follows.

        [q_x * i,  := sin(theta/2) * a_x  (where the vector 'a' is axis of rotation)
         q_y * j,  := sin(theta/2) * a_y
         q_z * k,  := sin(theta/2) * a_z
         q_w]      := cos(theta/2)        (& theta is the angle of rotation)
    """
    
    def __init__(
            self,
            quatVec: np.ndarray | None = None,
            inVersor: Versor | None = None,
            imagVec: np.ndarray | None = None,
            realPart: float | np.float32 | np.float64 | int | None = None,
            axisVec: np.ndarray = None,
            angle: float | np.float32 | np.float64 | int | None = None
    ):
        self._qvec: np.ndarray | None = None   # quaternion vector form: [q_x, q_y, q_z, q_w]^T
        self._axis = None   # axis of rotation: a 3-tuple
        self._halfAngle = None  # half the rotation angle
        
        if quatVec is not None:
            if all([isinstance(qc, NUMERIC_TYPE) for qc in quatVec]):
                try:
                    self._qvec = quatVec[:4]
                
                except IndexError:
                    raise TypeError("quatVec must be a sequence of length 4.")
            else:
                raise TypeError("quatVec must have integer or floating-point " \
                        + "components.")
        
        elif inVersor is not None:
            if inVersor._qvec is not None:
                self._qvec = inVersor._qvec[:]
            else:
                self._halfAngle = inVersor._halfAngle
                self._axis = np.array(inVersor._axis[:3])
        
        elif imagVec is not None and realPart is not None:
            # If all components of imagVec are supported numeric types
            if all([isinstance(i, NUMERIC_TYPE) for i in imagVec]):
                if isinstance(realPart, NUMERIC_TYPE):
                    try:
                        self._qvec = imagVec[:3] + [realPart]
                    except IndexError:
                        raise TypeError( \
                                "imagVec must be a sequence of length 3.")
                else:
                    raise TypeError("realPart must be integer or floating-point.")
            else:
                raise TypeError("imagVec must have integer or floating-point " \
                        + "components.")
        
        elif axisVec is not None and angle is not None:
            if all([isinstance(ac, NUMERIC_TYPE) for ac in axisVec]):
                if isinstance(angle, NUMERIC_TYPE):
                    try:
                        axisArray = np.array(axisVec[:3])
                        # Normalize the axis vector, so that it's a unit vector
                        axisNorm = la.norm(axisArray)
                        if axisNorm != 1.0:
                            axisArray /= axisNorm

                        self._axis = axisArray
                        self._halfAngle = angle / 2.0
                    except IndexError:
                        raise TypeError("axisVec must be a sequence of length 3.")
                else:
                    raise TypeError("angle must be an integer or floating-point.")
            else:
                raise TypeError("axisVec must have integer or floating-point " \
                    + "components.")
            
        else:
            raise TypeError("Invalid parameters.")
    
    @classmethod
    def fromAngleAxis(cls, angle, axisVector):
        # TODO: Check argument validity.
        return Versor(axisVec = axisVector, angle = angle)
    
    @classmethod
    def fromQuaternionVector(cls, quatVector):
        # TODO: Check argument validity.
        return Versor(quatVec = quatVector)
    
    @classmethod
    def fromVectorAndRealParts(cls, imagVector, realPart):
        # TODO: Check argument validity.
        return Versor(imagVec = imagVector, realPart = realPart)

    @classmethod
    def from_rotation_matrix(cls, rotationMatrix: np.ndarray):
        """
        Converts a 3x3 rotation matrix to a versor (unit quaternion) using one of four formulas, selected based on the
        maximum of the provided matrix's diagonal components and its trace, ensuring numerical stability.

        Note: This method uses the JPL notation for versors, where the imaginary components come first and the real
        component comes last (i.e., [q_x, q_y, q_z, q_w]^T).

        :param rotationMatrix: a 3x3 rotation matrix, as a 2D NumPy array.
        :return: a `Versor` object representing the same rotation as the provided rotation matrix.
        """
        trace = rotationMatrix.trace()

        diag1 = rotationMatrix[0][0]
        diag2 = rotationMatrix[1][1]
        diag3 = rotationMatrix[2][2]
        maxPivot = max(trace, diag1, diag2, diag3)

        # Use the most positive of the diagonal entries and the trace, as a
        # pivot, ensuring numerical stability:
        if maxPivot == diag1:
            r = math.sqrt(1.0 + diag1 - diag2 - diag3)
            s = 0.5 / r

            x = 0.5 * r
            y = (rotationMatrix[0][1] + rotationMatrix[1][0]) * s
            z = (rotationMatrix[2][0] + rotationMatrix[0][2]) * s
            w = (rotationMatrix[1][2] - rotationMatrix[2][1]) * s

            qvec: np.ndarray = np.array([x, y, z, w])
        elif maxPivot == diag2:
            r = math.sqrt(1.0 + diag2 - diag1 - diag3)
            s = 0.5 / r

            x = (rotationMatrix[0][1] + rotationMatrix[1][0]) * s
            y = 0.5 * r
            z = (rotationMatrix[1][2] + rotationMatrix[2][1]) * s
            w = (rotationMatrix[2][0] - rotationMatrix[0][2]) * s

            qvec: np.ndarray = np.array([x, y, z, w])
        elif maxPivot == diag3:
            r = math.sqrt(1.0 + diag3 - diag1 - diag2)
            s = 0.5 / r

            x = (rotationMatrix[0][2] + rotationMatrix[2][0]) * s
            y = (rotationMatrix[1][2] + rotationMatrix[2][1]) * s
            z = 0.5 * r
            w = (rotationMatrix[0][1] - rotationMatrix[1][0]) * s

            qvec: np.ndarray = np.array([x, y, z, w])
        else:  # o.w., maxPivot == trace
            r = math.sqrt(1.0 + trace)
            s = 0.5 / r

            x = (rotationMatrix[1][2] - rotationMatrix[2][1]) * s
            y = (rotationMatrix[2][0] - rotationMatrix[0][2]) * s
            z = (rotationMatrix[0][1] - rotationMatrix[1][0]) * s
            w = 0.5 * r

            qvec: np.ndarray = np.array([x, y, z, w])

        return Versor(quatVec=qvec)

    @classmethod
    def from_rotation_vector(cls, rotationVector):
        rot_vec: np.ndarray = np.asarray(rotationVector, dtype=float)

        if rot_vec.shape != (3,):
            raise TypeError("rot_vec must be a 3D vector.")

        angle = la.norm(rot_vec)

        if rot_vec[0] != 0 or rot_vec[1] != 0 or rot_vec[2] != 0:
            axisVec = rot_vec / angle
        else:
            return cls.unit()

        return Versor.fromAngleAxis(angle, axisVec)
    
    @classmethod
    def newPureImaginary(cls, imagVector):
        # TODO: Check argument validity.
        return Versor(imagVec = imagVector[:], realPart = 0.0)

    @classmethod
    def copy(cls, toCopy):
        if isinstance(toCopy, Versor):
            return Versor(inVersor = Versor)
        else:
             raise TypeError("toCopy must be a Versor.")
    
    @classmethod
    def unit(cls):
        return Versor(quatVec = (0.0,) * 3 + (1.0,))       # (q_i, q_j, q_k, q_r)
    
    @classmethod
    def zero(cls):
        return Versor(quatVec = (0.0,) * 4)               # (q_i, q_j, q_k, q_r)
    
    @classmethod
    def i(cls):
        return Versor(quatVec = (1.0,) + (0.0,) * 3)      # (q_i, q_j, q_k, q_r)
    
    @classmethod
    def j(cls):
        return Versor(quatVec = (0.0, 1.0) + (0.0,) * 2)  # (q_i, q_j, q_k, q_r)
    
    @classmethod
    def k(cls):
        return Versor(quatVec = (0.0,) * 2 + (1.0, 0.0))  # (q_i, q_j, q_k, q_r)
        
    @classmethod
    def rotationMatrixFromAngleAxis(cls, angle, axisVector):
        axisSkewM = Versor._genVectorSkewSymmetricMatrix(axisVector)
        cosine = math.cos(angle)
        return (cosine * np.identity(3)) - (math.sin(angle) * axisSkewM) \
                + ((1 - cosine) * np.outer(axisVector, axisVector))

    @classmethod
    def calcVectorCrossProductMatrix(cls, vector):
        return Versor._genVectorSkewSymmetricMatrix(vector)
        
    def rotationAngle(self):
        if self._halfAngle is not None:
            return 2.0 * self._halfAngle
        else:    
            return 2.0 * self._calcAngleAxis()[0]
    
    def rotationAxis(self):
        if self._axis is not None:
            return np.copy(self._axis)
        else:
            return np.copy(self._calcAngleAxis()[1])
    
    def imagVector(self):
        if self._qvec is not None:
            return np.array(self._qvec[:3])
        else:            
            return np.array(self._calcQuatVector()[:3])
    
    def realPart(self):
        if self._qvec is not None:
            return self._qvec[3]
        else:
            return self._calcQuatVector()[3]
    
    def quatVector(self):
        if self._qvec is not None:
            return self._qvec[:]
        else:
            return self._calcQuatVector()[:]
    
    def toRotationVector(self):
        return self.rotationAxis() * self.rotationAngle()

    # TODO: Port missing __mul__() and conj() methods from the `quatmath` library.
    def applyRotation(self, vectorToRotate):
        vecQ = Versor.newPureImaginary(vectorToRotate)
        return (self * vecQ * self).imagVector()  # FIXME: The right-hand side should be the conjugate of the versor.
        
    # Version: 2.0
    def to_rotation_matrix(self):
        """
        Converts this versor to a 3x3 rotation matrix, using the following formula:
            `R = (2 * q_w^2 - 1) * I - (2 * q_w) * [q_x, q_y, q_z]_x + 2 * [q_x, q_y, q_z]^T * [q_x, q_y, q_z]`

        Note: This uses the JPL notation for versors, where the imaginary components come first and the real component
        comes last (i.e., [q_x, q_y, q_z, q_w]^T).
        - For a unit quaternion, this method returns a 3x3 proper orthogonal rotation matrix with determinant +1.
        (For example, a positive 90° rotation about `+x` sends `+y` to `-z`, when applied to a column vector.)

        :return: a 3x3 rotation matrix as a 2D `numpy.ndarray`, which represents the same rotation as this versor.
        """
        imagVec = self.imagVector()
        scaledI = (2 * self._qvec[3] * self._qvec[3] - 1.0) * np.identity(3)
        imagOuter = np.outer(imagVec, imagVec)
        skewM = Versor._genVectorSkewSymmetricMatrix(imagVec)
        return scaledI - ((2 * self._qvec[3]) * skewM) + (2 * imagOuter)
            
    # Version: 1.0
    # Alternate method for calculating rotation matrix (slightly more expensive).
#    def toRotationMatrix(self):
#        xiMatrix = _calcXiMatrix()
#        psiMatrix = _calcPsiMatrix()
#        return la.dot(xiMatrix.T, psiMatrix)
    
    def _calcPsiMatrix(self):
        imagVec = self.imagVector()
        realPart = self.realPart()
        skewM = Versor._genVectorSkewSymmetricMatrix(imagVec)
        diffM = (realPart * np.identity(3)) - skewM
        qVecH = imagVec.reshape(1, 3)

        return np.bmat([[diffM],
                        [-qVecH]])
                          
    def _calcXiMatrix(self):
        imagVec = self.imagVector()
        realPart = self.realPart()
        skewM = Versor._genVectorSkewSymmetricMatrix(imagVec)
        sumM = (realPart * np.identity(3)) + skewM
        qVecH = imagVec.reshape(1, 3)

        return np.bmat([[sumM],
                        [-qVecH]])

    def _calcQuatLeftMultiplyMatrix(self):
        """
        Generates the quaternion left-multiply matrix for this Versor, which
        when multiplied with a 4-vector quat-vector performs quaternion left-
        multiplication, as follows.
            qMat = q._genLeftMultiplyMatrix()
            r = np.array([[a, b, c]].T
            newQ = qMat * r     # matrix-vector multiplication

            newQ2 = q x (r, 0)  # pseudo-code: quaternion "left"-multiplication equivalent

            newQ == newQ2

        Note (Special Case):
            Given a rotation vector r and Versor v:

            r = [a, b, c]^T
            v = Versor.fromQuaternionVector([a, b, c, 0])
            rotM1 = v._genLeftMultiplyMatrix()
            rotM2 = Versor._genLeftSkewSymmetricMatrix(r)

            rotM1 == rotM2

        :return: the quaternion left-multiply matrix for this Versor.
        """
        psiM = self._calcPsiMatrix()
        quatVecV = np.array([self._qvec]).T

        return np.bmat([ [psiM, quatVecV] ])

    
    def _calcQuatRightMultiplyMatrix(self):
        """
        Generates the quaternion right-multiply matrix for this Versor, which
        when multiplied with a 4-vector quat-vector performs quaternion right-
        multiplication, as follows.
            qMat = q._genRightMultiplyMatrix()
            r = np.array([[a, b, c]].T
            newQ = qMat * r     # matrix-vector multiplication

            newQ2 = (r, 0) x q  # pseudo-code: quaternion "right"-multiplication equivalent

            newQ == newQ2

        Note (Special Case):
            Given a rotation vector r and Versor v:

            r = [a, b, c]^T
            v = Versor.fromQuaternionVector([a, b, c, 0])
            rotM1 = v._genRightMultiplyMatrix()
            rotM2 = Versor._genRightSkewSymmetricMatrix(r)

            rotM1 == rotM2

        :return: the quaternion right-multiply matrix for this Versor.
        """
        xiM = self._calcXiMatrix()
        quatVecV = np.array([self._qvec]).T

        return np.bmat([ [xiM, quatVecV] ])


    def _calcAngleAxis(self):
        if self._halfAngle is None:
            imagVec = np.array(self._qvec[:3])
            imagNorm = la.norm(imagVec)
            self._halfAngle = math.atan2(imagNorm, self._qvec[3])
            self._axis = imagVec / imagNorm

        return (self._halfAngle, self._axis)
    
    def _calcQuatVector(self):
        if self._qvec is None:
            imagVec = math.sin(self._halfAngle) * self._axis
            realPart = math.cos(self._halfAngle)
            self._qvec = (imagVec[0], imagVec[1], imagVec[2], realPart)
        
        return self._qvec

    @classmethod
    def _genLeftSkewSymmetricMatrix(cls, rotationVector):
        """
        Generates the quaternion left-multiply skew-symmetric matrix, from the
        given rotation vector. This function is often denoted by a capital
        omega in quaternion literature.
        :param rotationVector: a rotation vector from which to form the
            left-multiply matrix, supplied as a 1D numpy.array.
        :return: the quaternion left-multiply matrix for the given rotation
            vector.

        The matrix's structures is as follows:

            [ -[rotationVector]_X  rotationVector ]
            [ -rotationVector^T    0              ] ,

        where [vec]_X is the cross-product matrix for the vector 'vec', and
              vec^T is the transpose of the vector 'vec'.
        """
        vecSkewM = Versor._genVectorSkewSymmetricMatrix(rotationVector)
        rotVec = np.array([rotationVector]).T

        return np.bmat([ [ -vecSkewM, rotVec ],\
                         [ -rotVec.T, np.zeros((1,1)) ] ])

        # return np.array([ np.append(-vecSkewM[0], rotationVector[0]), \
        #                   np.append(-vecSkewM[1], rotationVector[1]), \
        #                   np.append(-vecSkewM[2], rotationVector[2]), \
        #                   [ -rotationVector[0], -rotationVector[1], -rotationVector[2], 0 ] ])


    @classmethod
    def _genRightSkewSymmetricMatrix(cls, rotationVector):
        """
        Generates the quaternion right-multiply skew-symmetric matrix, from the
        given rotation vector. This function is often denoted by a capital
        lambda in quaternion literature.
        :param rotationVector: a rotation vector from which to form the
            right-multiply matrix.
        :return: the quaternion right-multiply matrix for the given rotation
            vector.

        The matrix's structures is as follows:

            [ [rotationVector]_X  rotationVector ]
            [ -rotationVector^T   0              ] ,

        where [vec]_X is the cross-product matrix for the vector 'vec', and
              vec^T is the transpose of the vector 'vec'.
        """
        vecSkewM = Versor._genVectorSkewSymmetricMatrix(rotationVector)
        rotVec = np.array([rotationVector]).T

        return np.bmat([ [ vecSkewM, rotVec ],\
                         [ -rotVec.T, np.zeros((1,1)) ] ])


    @classmethod
    def _genVectorSkewSymmetricMatrix(cls, vector):
        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])
