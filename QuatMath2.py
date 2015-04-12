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


# Global Constants:

DEF_THRESHOLD = 1.0e-10  # Default threshold for equality comparisons
NUMERIC_TYPE = (int, long, float, np.float32, np.float64)


class Versor(object):
    
    def __init__(self, \
                 quatVec = None, \
                 inVersor = None, \
                 imagVec = None, \
                 realPart = None, \
                 axisVec = None, \
                 angle = None):
        
        self._qvec = None   # quaternion vector form: [q_i, q_j, q_k, q_r]^T
        self._axis = None   # axis of rotation: a 3-tuple
        self._halfAngle = None  # half the rotation angle
        
        if quatVec != None:
            if all([isinstance(qc, NUMERIC_TYPE) for qc in quatVec]):
                try:
                    self._qvec = quatVec[:4]
                
                except IndexError:
                    raise TypeError("quatVec must have integer or " \
                        + "floating-point components.")
            else:
                raise TypeError("quatVec must be a sequence of length 4.")
        
        elif inVersor != None:
            if inVersor._qvec != None:
                self._qvec = inVersor._qvec[:]
            else:
                self._halfAngle = inVersor._halfAngle
                self._axis = inVersor._axis
        
        elif imagVec != None and realPart != None:
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
        
        elif axisVec != None and angle != None:
            if all([isinstance(ac, NUMERIC_TYPE) for ac in axisVec]):
                if isinstance(angle, NUMERIC_TYPE):
                    try:
                        self._axis = axisVec[:3]
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
        return Versor(axisVec = axisVector[:], angle = angle)
    
    @classmethod
    def fromQuaternionVector(cls, quatVector):
        # TODO: Check argument validity.
        return Versor(quatVec = quatVector[:])
    
    @classmethod
    def fromVectorAndRealParts(cls, imagVector, realPart):
        # TODO: Check argument validity.
        return Versor(imagVec = imagVector[:], realPart = realPart)
    
    @classmethod
    def fromRotationVector(cls, rotationVector):
        # TODO: Further check argument validity.
        rotVec = None
        if not isinstance(rotationVector, np.array):
            rotVec = np.array(rotationVector)
        
        angle = la.norm(rotVec)
        axisVec = rotVec / angle
        
        return Versor.fromAngleAxis(angle, axisVec)
    
    @classmethod
    def newPureImaginary(cls, imagVector):
        # TODO: Check argument validity.
        return Versor(imagVec = imagVector[:], realPart = 0.0)

    @classmethod
    def copy(cls, toCopy):
        if isinstance(toCopy, versor):
            return Versor(inVersor = versor)
        else:
             raise TypeError("toCopy must be a Versor.")
    
    @classmethod
    def unit(cls):
        return Versor(quatVec = (0.0,) * 3 + (1.0))       # (q_i, q_j, q_k, q_r)
    
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
    
    
    def rotationAngle(self):
        if self._angle != None:
            return 2.0 * self._angle
        else:    
            return 2.0 * self._calcAngleAxis()[0]
    
    def rotationAxis(self):
        if self._axis != None:
            return np.copy(self._axis)
        else:
            return np.copy(self._calcAngleAxis()[1])
    
    def imagVector(self):
        if self._qvec != None:
            return np.array(self._qvec[:3])
        else:            
            return np.array(self._calcQuatVector()[:3])
    
    def realPart(self):
        if self._qvec != None:
            return self._qvec[3]
        else:
            return self._calcQuatVector()[3]
    
    def quatVector(self):
        if self._qvec != None:
            return self._qvec[:]
        else:
            return self._calcQuatVector()[:]
    
    def toRotationVector(self):
        return self.rotationAxis() * self.rotationAngle()
        
    def applyRotation(self, vectorToRotate):
        vecQ = Versor.newPureImaginary(vectorToRotate)
        return (self * vecQ * self).imagVector()
    
    def _calcAngleAxis(self):
        if self._angle == None:
            imagVec = np.array(self._qvec[:3])
            imagNorm = la.norm(imagVec)
            self._angle = math.atan2(imagNorm, self._qvec[3])
            self._axis = imagVec / imagNorm
                        
        return self._angle, self._axis
    
    def _calcQuatVector(self):
        if self._qvec == None:
            imagVec = math.sin(self._halfAngle) * self._axis
            realPart = math.cos(self._halfAngle)
            self._qvec = (imagVec[0], imagVec[1], imagVec[2], realPart)
        
        return self._qvec
    