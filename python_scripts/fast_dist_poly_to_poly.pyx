import cython
cimport cpython.array

import numpy as np
cimport numpy as np

cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    char GEOSContains_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil

cdef GEOSContextHandle_t get_geos_context_handle():
    # Note: This requires that lgeos is defined, so needs to be imported as:
    from shapely.geos import lgeos
    cdef np.uintp_t handle = lgeos.geos_handle
    return <GEOSContextHandle_t>handle


@cython.boundscheck(False)  # won't check that index is in bounds of array
@cython.wraparound(False) # array[-1] won't work
cpdef distance(self, other):
    cdef int n = self.size
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom = other.__geom__  # a geometry pointer
    geometries = self._geometry_array
    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> geometries[idx]
            if left_geom != NULL:
                distance = GEOSDistance_r(left_geom, some_point.__geom)
            else:
                distance = NaN
    return(distance)
