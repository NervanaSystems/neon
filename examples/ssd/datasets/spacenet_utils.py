from osgeo import gdal, osr, ogr
import numpy as np


def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    # type: (object, object, object, object, object) -> object

    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(lon, lat)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    # print(x_origin)
    y_origin = transform[3]
    # print(y_origin)
    pixel_width = transform[1]
    # print(pixel_width)
    pixel_height = transform[5]
    # print(pixel_height)
    geom.Transform(coord_trans)
    # print(geom.GetPoint())
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)


def geoPolygonToPixelPolygonWKT(geom, inputRaster, targetSR, geomTransform,
                                breakMultiPolygonGeo=True, pixPrecision=2):
    # Returns Pixel Coordinate List and GeoCoordinateList

    polygonPixBufferList = []
    polygonPixBufferWKTList = []
    if geom.GetGeometryName() == 'POLYGON':
        polygonPix = ogr.Geometry(ogr.wkbPolygon)
        for ring in geom:
            # GetPoint returns a tuple not a Geometry
            ringPix = ogr.Geometry(ogr.wkbLinearRing)

            for pIdx in range(ring.GetPointCount()):
                lon, lat, z = ring.GetPoint(pIdx)
                xPix, yPix = latlon2pixel(lat, lon, inputRaster, targetSR, geomTransform)

                xPix = round(xPix, pixPrecision)
                yPix = round(yPix, pixPrecision)
                ringPix.AddPoint(xPix, yPix)

            polygonPix.AddGeometry(ringPix)
        polygonPixBuffer = polygonPix.Buffer(0.0)
        polygonPixBufferList.append([polygonPixBuffer, geom])

    elif geom.GetGeometryName() == 'MULTIPOLYGON':

        for poly in geom:
            polygonPix = ogr.Geometry(ogr.wkbPolygon)
            for ring in poly:
                # GetPoint returns a tuple not a Geometry
                ringPix = ogr.Geometry(ogr.wkbLinearRing)

                for pIdx in range(ring.GetPointCount()):
                    lon, lat, z = ring.GetPoint(pIdx)
                    xPix, yPix = latlon2pixel(lat, lon, inputRaster, targetSR, geomTransform)

                    xPix = round(xPix, pixPrecision)
                    yPix = round(yPix, pixPrecision)
                    ringPix.AddPoint(xPix, yPix)

                polygonPix.AddGeometry(ringPix)
            polygonPixBuffer = polygonPix.Buffer(0.0)
            if breakMultiPolygonGeo:
                polygonPixBufferList.append([polygonPixBuffer, poly])
            else:
                polygonPixBufferList.append([polygonPixBuffer, geom])

    for polygonTest in polygonPixBufferList:
        if polygonTest[0].GetGeometryName() == 'POLYGON':
            polygonPixBufferWKTList.append([polygonTest[0].ExportToWkt(),
                                            polygonTest[1].ExportToWkt()])
        elif polygonTest[0].GetGeometryName() == 'MULTIPOLYGON':
            for polygonTest2 in polygonTest[0]:
                polygonPixBufferWKTList.append([polygonTest2.ExportToWkt(),
                                                polygonTest[1].ExportToWkt()])

    return polygonPixBufferWKTList


def get_bounding_boxes(img_file, annot_file):

    srcRaster = gdal.Open(img_file)
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(srcRaster.GetProjectionRef())
    geomTransform = srcRaster.GetGeoTransform()

    dataSource = ogr.Open(annot_file, 0)
    layer = dataSource.GetLayer()

    building_id = 0
    buildinglist = []

    for feature in layer:
        geom = feature.GetGeometryRef()
        geom_wkt_list = geoPolygonToPixelPolygonWKT(geom, img_file, targetSR, geomTransform)

        for geom_wkt in geom_wkt_list:
            building_id += 1
            buildinglist.append(ogr.CreateGeometryFromWkt(geom_wkt[0]).GetEnvelope())

    return buildinglist


def load_as_uint8(filename):

    image = gdal.Open(filename)
    image_array = np.array(image.ReadAsArray())

    image_uint8 = np.zeros(image_array.shape, dtype=np.uint8)
    # rescale each band to be between 0, 255

    for k, band in enumerate(image_array):
        band_max = np.max(band)

        if band_max != 0:
            band = band.astype(np.float) / band_max * 255.0

        image_uint8[k, :, :] = band

    return image_uint8
