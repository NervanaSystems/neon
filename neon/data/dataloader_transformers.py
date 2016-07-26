import numpy as np

from neon import NervanaObject


class DataLoaderTransformer(NervanaObject):
    """
    DataLoaderTransformers are used to transform the output of a DataLoader.
    DataLoader doesn't have easy access to the device or graph, so any
    computation that should happen there should use a DataLoaderTransformer.
    """
    def __init__(self, dataloader, index=None):
        super(DataLoaderTransformer, self).__init__()

        self.dataloader = dataloader
        self.index = index

        if self.index is not None:
            # input shape is contiguous
            data_size = reduce(
                lambda x, y: x * y,
                self.dataloader.shapes()[index],
            )

            self._shape = (data_size, self.be.bsz)

    def __getattr__(self, key):
        return getattr(self.dataloader, key)

    def __iter__(self):
        for tup in self.dataloader:
            if self.index is None:
                yield self.transform(tup)
            else:
                ret = self.transform(tup[self.index])
                if ret is None:
                    raise ValueError(
                        '{} returned None from a transformer'.format(
                            self.__class__.__name__
                        )
                    )

                out = list(tup)
                out[self.index] = ret
                yield out

    def transform(self, t):
        raise NotImplemented()


class OneHot(DataLoaderTransformer):
    """
    OneHot will convert `index` into a onehot vector.
    """
    def __init__(self, dataloader, nclasses, *args, **kwargs):
        super(OneHot, self).__init__(dataloader, *args, **kwargs)

        self.output = self.be.empty((nclasses, self.be.bsz))

    def transform(self, t):
        self.output[:] = self.be.onehot(t, axis=0)
        return self.output


class TypeCast(DataLoaderTransformer):
    """
    TypeCast data from dataloader at `index` to dtype and move into
    device memory if not already.
    """
    def __init__(self, dataloader, index, dtype, *args, **kwargs):
        super(TypeCast, self).__init__(
            dataloader, index=index, *args, **kwargs
        )

        self.output = self.be.empty(self._shape, dtype=dtype)

    def transform(self, t):
        self.output[:] = t
        return self.output


class ImageMeanSubtract(DataLoaderTransformer):
    """
    subtract pixel_mean from data at `index`.  Assumes data is in CxHxWxN
    """
    def __init__(self, dataloader, index, pixel_mean, *args, **kwargs):
        super(ImageMeanSubtract, self).__init__(
            dataloader, index=index, *args, **kwargs
        )

        pixel_mean = np.asarray(pixel_mean)
        self.pixel_mean = self.be.array(pixel_mean[:, np.newaxis])

    def transform(self, t):
        # create a view of t and modify that.  Modifying t directly doesn't
        # work for some reason ...
        tr = t.reshape((3, -1))
        tr[:] = tr - self.pixel_mean
        return t


class DumpImage(DataLoaderTransformer):
    def __init__(self, dataloader, index, image_index, outshape,
                 output_directory=None, *args, **kwargs):
        """
        dump image number `image_index` in data `index` to a random
        file in `output_directory`.
        """
        super(DumpImage, self).__init__(
            dataloader, index=index, *args, **kwargs
        )
        self.outshape = outshape
        self.image_index = image_index
        self.output_directory = output_directory or '/tmp'
        if self.output_directory[-1] != '/':
            self.output_directory += '/'

    def transform(self, t):
        # grab one image from t, and bring it into host mem
        if isinstance(t, np.ndarray):
            a = t
        else:
            a = t.get()

        a = a[:, self.image_index]
        # coming from DataLoader first dimensioned has been flattened
        a = a.reshape(self.outshape)
        # transpose from CHW to H W C
        a = a.transpose(1, 2, 0)
        # reorder color channel
        a = a[:, :, ::-1]
        # TODO: see if this can be removed.
        a = a.astype('uint8')

        from PIL import Image as PILImage
        img = PILImage.fromarray(a)
        img.save(self.filename())

        # return unmodified tensor
        return t

    def filename(self):
        """
        generate random filename
        """
        import random
        return self.output_directory + str(random.random()) + '.png'



