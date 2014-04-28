Finding stars
=============

The following functions are provided to find stars in an image:

* `daofind(data, fwhm, threshold, ....)`
* `starfind(data, fwhm, threshold, ....)`
* `epsffind(data, psf, threshold, ....)`   (work in progress)


Examples
--------

Create an image with a single 2D circular Gaussian source to represent
a star and find it in the image using ``daofind``:

  >>> import numpy as np
  >>> import findobj
  >>> y, x = np.mgrid[-50:51, -50:51]
  >>> img = 100.0 * np.exp(-(x**2/5.0 + y**2/5.0))
  >>> tbl = findobj.daofind(img, 3.0, 1.0)
  >>> tbl.pprint(max_width=-1)
    id xcen ycen     sharp      round1 round2 npix sky  peak      flux          mag
   --- ---- ---- -------------- ------ ------ ---- --- ----- ------------- --------------
     1 50.0 50.0 0.440818817057    0.0    0.0 25.0 0.0 100.0 62.4702758896 -4.48918355985


Search the same image, but using ``irafstarfind``:

  >>> tbl2 = findobj.irafstarfind(img, 3.0, 1.0)
  >>> tbl2.pprint(max_width=-1)
    id xcen ycen      fwhm         sharp            round             pa      npix      sky           peak          flux          mag
   --- ---- ---- ------------- -------------- ----------------- ------------- ---- ------------- ------------- ------------- --------------
     1 50.0 50.0 2.04509092195 0.681696973984 7.36564629863e-17 178.865861588 13.0 31.2551800113 68.7448199887 469.034565146 -6.67801212224


Create an image with three 2D circular Gaussian source to represent
stars, find them in the image using ``daofind``, and display the
results in a browser with interactive searching and sorting:

  >>> import numpy as np
  >>> import findobj
  >>> y, x = np.mgrid[0:101, 0:101]
  >>> img = 100.0 * np.exp(-((x-50)**2/5.0 + (y-50)**2/5.0))
  >>> img += 250.0 * np.exp(-((x-65.2)**2/4.0 + (y-75.9)**2/4.0))
  >>> img += 500.0 * np.exp(-((x-30.78)**2/3.0 + (y-25.313)**2/3.2))
  >>> tbl = findobj.daofind(img, 3.0, 1.0)
  >>> tbl.show_in_browser(jsviewer=True)

The three sources should be centered at ``(x, y) = (50, 50), (65.2, 75.9),
and (30.78, 25.313)``.  Now display the image and mark the location
of the found sources:

  >>> import matplotlib.pyplot as plt
  >>> plt.imshow(img, cmap=plt.cm.Greys)
  >>> plt.scatter(tbl['xcen'], tbl['ycen'], s=800, color='cyan', facecolor='none')
  >>> plt.show()


.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import findobj
    y, x = np.mgrid[0:101, 0:101]
    img = 100.0 * np.exp(-((x-50)**2/5.0 + (y-50)**2/5.0))
    img += 250.0 * np.exp(-((x-65.2)**2/4.0 + (y-75.9)**2/4.0))
    img += 500.0 * np.exp(-((x-30.78)**2/3.0 + (y-25.313)**2/3.2))
    tbl = findobj.daofind(img, 3.0, 1.0)
    fig = plt.imshow(img, vmax=200.0, origin='lower',
        extent=(0, 100, 0, 100))
    fig.set_cmap('hot')
    plt.scatter(tbl['xcen'], tbl['ycen'], s=800, color='cyan',
        facecolor='none')
    plt.axis('off')
    plt.show()


Finally, filter the catalog to include only sources with a peak flux > 200
(resulting in only two sources):

  >>> newtbl = tbl[tbl['peak'] > 200]
  >>> newtbl.show_in_browser(jsviewer=True)
  >>> newtbl.pprint(max_width=-1)
    id      xcen          ycen         sharp           round1           round2      npix sky      peak          flux          mag
   --- ------------- ------------- -------------- ---------------- ---------------- ---- --- ------------- ------------- --------------
     1 30.7757703041 25.3263301704 0.477860513808 -0.0683222486336  0.0704298851828 25.0 0.0 477.163620787 371.207549568 -6.42404200065
     3 65.2042829915 75.8989787037 0.456567416754 -0.0385405609864 -0.0120707450026 25.0 0.0 246.894450123  173.36836323 -5.59742462258



