class PSFPhotometry(object):
    """
    This is an implementation of the NSTAR algorithm proposed by Stetson
    (1987) to perform point spread function photometry in crowded fields.

    This implementation basically relys on the loop FIND, GROUP, NSTAR,
    SUBTRACT, FIND until no more stars are detected.
    """

    def __init__(self, find, group, bkg, psf_model, fitter):
        """
        Parameters
        ----------

        find : an instance of any StarFinderBase subclasses
        group : an instance of any GroupStarsBase subclasses
        bkg : an instance of any BackgroundBase2D (?) subclasses
        psf_model : Fittable2DModel instance
        fitter : Fitter instance
        """

        self.find = find
        self.group = group
        self.bkg = bkg
        self.psf_model = psf_model
        self.fitter = fitter

    @property
    def find(self):
        return self._find
    
    @find.setter
    def find(self, find):
        if isinstance(find, StarFinderBase)
            self._find = find
        else:
            raise ValueError('find is expected to be an instance of '
                             'StarFinderBase, received {}'.format(type(find)))
    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, group):
        if isinstance(group, GroupStarsBase):
            self._group = group
        else:
            raise ValueError('group is expected to be an instance of '
                             'GroupStarsBase, received {}.'.\
                             format(type(group)))
    
    @property
    def bkg(self)
        return self.bkg

    @background.setter
    def bkg(self, bkg):
        if isinstance(bkg, BackgroundBase2D):
            self._background = background
        else:
            raise ValueError('bkg is expected to be an instance of '
                             'BackgroundBase2D, received {}.'\
                             .format(type(bkg)))
    
    @property
    def psf_model(self):
        return self._psf_model

    @psf_model.setter
    def psf_model(self, psf_model):
        if isinstance(psf_model, Fittable2DModel):
            self._psf_model = psf_model
        else:
            raise ValueError('psf_model is expected to be an instance of '
                             'Fittable2DModel, received {}.'\
                             .format(type(fitter)))

    @property
    def fitter(self):
        return self._fitter

    @fitter.stter
    def fitter(self, fitter):
        if isinstance(fitter, Fitter):
            self._fitter = fitter
        else:
            raise ValueError('fitter is not a valid astropy Fitter, '
                             'received {}'.format(type(fitter)))

    def nstar(self, image, groups, fitshape, bkg, psf_model, fitter):
        """
        Perform the simultaneous profile fitting of the sources in ``groups``.
        """

    
    def perform_photometry(self, image, niters, fitshape):
        """
        Parameters
        ----------
        image : array-like, ImageHDU, HDUList
            image to perform photometry
        niters : int
            number of iterations for the loop FIND, GROUP, SUBTRACT, NSTAR
        fitshape : tuple
            rectangular shape around the center of a star which will be used
            to collect the data to do the fitting

        Returns
        -------
        outtab : astropy.table.Table
            Table with the photometry results, i.e., centroids and flux
            estimations.
        residual_image : array-like, ImageHDU, HDUList
            Residual image calculated by subtracting the fitted sources
            and the original image
        """
        
        # prepare output table
        outtab = Table([[], [], [], [], []],
                       names=('id', 'x_fit', 'y_fit', 'flux_fit',
                              'iter_detected'),
                       dtype=('i4','f8','f8','f8','i4'))

        # make a copy of the input image
        residual_image = image.copy()

        # find potential sources on the given image
        sources = self.find(residual_image)

        n = 1
        # iterate until no more sources are found or the number of iterations
        # has been reached
        while(n <= niters and len(sources) > 0):
            # prepare input table
            intab = Table(names=['id', 'x_0', 'y_0', 'flux_0'],
                          data=[sources['id'], sources['xcentroid'],
                          sources['ycentroid'], sources['flux']])

            # find groups of overlapping sources
            groups = self.group(intab)

            # fit the sources within in each group in a simultaneous manner
            # and get the residual image
            curr_tab, residual_image = self.nstar(residual_image, groups,
                                                  fitshape, self.bkg,
                                                  self.psf_model, self.fitter)

            # marks in which iteration those sources were fitted
            curr_tab['iter_detected'] = n

            # populate output table
            outtab = vstack([outtab, curr_tab])
            
            # find remaining sources in the residual image
            sources = self.find(residual_image)
            n += 1

        return outtab, residual_image
