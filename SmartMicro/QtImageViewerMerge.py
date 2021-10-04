""" Implements a merge viewer for different channels in the same ViewBox. Used in NNGui and
NetworkWatchdog.

    self.viewerProc = QtImageViewerMerge()
    self.imageItemDrpProc = self.viewerProc.addImage()
    self.imageItemMitoProc = self.viewerProc.addImage()
    self.viewerProc.setLUT(self.imageItemDrpProc, 'reds')
    self.viewerProc.setLUT(self.imageItemMitoProc, 'grey')

    A custom LUT can be added in LUTItemSimple by adding to the dict.
            self.customGradients = pg.OrderedDict([('reds', {'ticks': [(0.0, (0, 0, 0, 255)),
                                                            (1.0, (255, 0, 0, 255))],
                                                'mode': 'hsv'})])

    This is a more 'leight weight' approach to what pg.ImageViewer implements with the
    HistogramLutItem and many things are copied from there.

Returns:
    [type]: [description]
"""

import sys

import numpy as np
import PyQt5.QtCore as QtCore
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPainter, QPicture
from PyQt5.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
                             QMainWindow, QPushButton, QSizePolicy, QSlider,
                             QWidget)
from skimage import io


class CrossItem(pg.GraphicsObject):
    """ A cross that can be added to a pg.ViewBox to point """
    def __init__(self, rect, color='#CCCCCC', parent=None):

        super().__init__(parent)
        self._rect = rect
        self.color = color
        self.picture = QPicture()
        self.generatePicture()
        self.setZValue(100)

    @property
    def rect(self):
        """ original rectangle give at initialization """
        return self._rect

    def generatePicture(self, pos=(0, 0)):
        """ Generate the cross. The size should at some point be set depending on the Range of
        the ViewBox for example to ensure always the same apparent size. """
        size = 5
        painter = QPainter(self.picture)
        painter.setPen(pg.mkPen(color=self.color, width=2))
        painter.setBrush(pg.mkBrush(color=self.color))
        painter.drawLine(pos[0]-size, pos[1]-size, pos[0]+size, pos[1]+size)
        painter.drawLine(pos[0]+size, pos[1]-size, pos[0]-size, pos[1]+size)
        painter.end()

    def setPosition(self, pos):
        """ set the Position of the cross after translating to a QPointF"""
        pos = QPointF(pos[0][1]+0.5, pos[0][0]+0.5)
        self.setPos(pos)

    def paint(self, *painter):
        """ I think this is used by the ViewBox when displaying the cross """
        painter[0].drawPicture(0, 0, self.picture)

    def boundingRect(self):
        """ Bounding rectangule of the cross as QRectF """
        return QRectF(self.picture.boundingRect())


class QtImageViewerMerge(QMainWindow):  # GraphicsWindow):
    """ Main widget of this module that can be used to display two or more channels in a merge view,
    or just one channel to use the LUT capabilities that are built in. Leight weight and fast for
    GUIs with many instances of this or ones that have to be as fast as possible. """
    frameChanged = pyqtSignal([], [int])
    resizedEmitter = pyqtSignal([], [int])

    def __init__(self, maxRange=1):
        QMainWindow.__init__(self)
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.glw = pg.GraphicsLayoutWidget()

        self.layoutBox = QGridLayout(self.widget)
        self.layoutBox.setContentsMargins(0, 0, 0, 0)
        self.layoutBox.addWidget(self.glw)

        # Image frame viewer.
        self.viewBox = self.glw.addViewBox()
        self.viewBox.setAspectLocked()
        self.viewBox.invertY()
        self.cross = CrossItem(QRectF(0, 0, 1, 1), color='#BBBBBB')
        self.viewBox.addItem(self.cross)
        self.crossRational = CrossItem(QRectF(0, 0, 1, 1), color='#02CFBB')
        self.viewBox.addItem(self.crossRational)

        # add a Menu on top that can be toggled with a button
        self.toggleMenu = QFrame(self.widget)
        self.gridMenu = QGridLayout(self.toggleMenu)
        self.menuButton = QPushButton("...", self.widget)
        self.menuButton.setFixedSize(20, 20)
        self.gridMenu.addWidget(self.menuButton, 0, 0, Qt.AlignTop)
        self.viewBox.sigRangeChanged.connect(self.rangeChanged)

        self.maxRange = maxRange
        self.zValue = 0
        self.fullImages = []
        self.imageItems = []
        self.saturationSliders = []
        self.opacitySliders = []
        self.qFrames = []
        self.numChannels = 0
        self.toggle = 0
        self.menuButton.clicked.connect(self.showMenu)
        # layoutBox = QHBoxLayout(self.widget)
        # self.viewBox.sigRangeChanged.connect(self.resizedEvent)

    def rangeChanged(self):
        """ Crop the image so only the part that is visible is actually added to the ViewBox """
        for pos in range(self.numChannels):
            if self.fullImages[pos] is None:
                return
            visRect = self.viewBox.viewRect()
            fullSize = len(self.fullImages[pos])
            oldLevels = self.imageItems[pos]['ImageItem'].getLevels()
            adjustedRect = visRect

            # Keep always one more pixel than necessary of the image on all sides
            adjustedRect = QRectF(round(adjustedRect.left())-1, round(adjustedRect.top())-1,
                                  round(adjustedRect.width())+2, round(adjustedRect.height())+2)

            # Adjust for edges of the image entering the ViewBox
            if visRect.bottom() > fullSize:
                adjustedRect.setBottom(fullSize)
            if visRect.right() > fullSize:
                adjustedRect.setRight(fullSize)
            if visRect.top() < 0:
                adjustedRect.setTop(0)
            if visRect.left() < 0:
                adjustedRect.setLeft(0)

            img = self.cropImagetoBox(self.fullImages[pos], adjustedRect)
            self.imageItems[pos]['ImageItem'].setImage(img, levels=oldLevels)
            adjustedRect = QRectF(round(adjustedRect.left()), round(adjustedRect.top()),
                                  round(adjustedRect.width()), round(adjustedRect.height()))

            self.imageItems[pos]['ImageItem'].setRect(adjustedRect)

    def cropImagetoBox(self, img, Rect):
        """ Crops image to the QRectF provided """
        img = img[round(Rect.top()):round(Rect.bottom())][:, round(Rect.left()):round(Rect.right())]
        return img

    def setImage(self, img, pos=0):
        """ set a new image img to the channel defined in pos and update the saturation Sliders """
        self.fullImages[pos] = img
        self.rangeChanged()

    def resetRanges(self):
        """ reset Ranges when a new stack is loaded in by some GUI """
        for pos in range(self.numChannels):
            fullImageRange = [np.min(self.fullImages[pos]), np.max(self.fullImages[pos])]
            maxImage = np.max([fullImageRange[1], self.maxRange])
            minImage = fullImageRange[0]
            self.saturationSliders[pos].viewBox.setYRange(-0.2*maxImage, maxImage*1.2)
            self.saturationSliders[pos].regions[0].setRegion((-0.1*maxImage, maxImage*1.1))
            self.imageItems[pos]['ImageItem'].setLevels((minImage, maxImage))

    def resetZoom(self):
        if self.fullImages[-1] is not None:
            self.viewBox.setRange(xRange=(0, self.fullImages[-1].shape[0]),
                                  yRange=(0, self.fullImages[-1].shape[1]))

    def addImage(self, img=None):
        """ Add an image item/channel on top of the other images that are already present. Do not
        init the alpha channel if its the first imageItem. Leave empty if no image is given  """
        imgItem = pg.ImageItem()
        imgItem.setAutoDownsample(True)
        self.viewBox.addItem(imgItem)
        imgItem.setZValue(self.zValue)
        self.zValue = self.zValue + 1
        imgStruct = {'ImageItem': imgItem, 'cm_name': None, 'transparency': False,
                     'opacity': (0.3, 0.85), 'saturation': 255}
        self.imageItems.append(imgStruct)
        self.fullImages.append(img)
        imgItem.setOpts(axisOrder='row-major')
        if self.numChannels > 0:
            imgItem.setCompositionMode(QPainter.CompositionMode_Screen)
        self.addMenu()
        if img is not None:
            self.setImage(img, self.numChannels)
        self.numChannels = self.numChannels + 1
        self.resetZoom()
        return imgItem

    def setLUT(self, img, name='hot'):
        """ set the lookuptable from the outside without using the popup context menu """
        for i in range(0, len(self.imageItems)):
            if self.imageItems[i]['ImageItem'] == img:
                channel = i
        self.saturationSliders[channel].loadPresetLUT(name)

        self.updateImage(channel)
        # self.imageItems[channel]['cm_name'] = name
        # self.imageItems[channel]['transparency'] = transparency
        # self.imageItems[channel]['opacity'] = opacity
        # colormap = getImageItemcolormap(name, transparency, opacity)
        # img.setLookupTable(colormap)

    def resizedEvent(self):
        """ Emit that you have been resized """
        self.resizedEmitter.emit()

    def addMenu(self):
        """" Add a menu to the channel for LUT and opacity regulations """
        channel = self.numChannels
        thisMenu = QFrame(self.toggleMenu)
        self.qFrames.append(thisMenu)
        thisMenuLayout = QGridLayout(thisMenu)
        self.saturationSliders.append(LUTItemSimple())
        self.saturationSliders[self.numChannels].gradientChanged.connect(
            lambda: self.updateImage(channel))
        self.saturationSliders[self.numChannels].levelChanged.connect(
            lambda: self.adjustLevel(channel))
        if channel > 0:
            self.saturationSliders[self.numChannels].regions[0].setRegion((100, 255))

        # self.saturationSliders[self.numChannels].setValue(255)
        thisMenuLayout.addWidget(self.saturationSliders[self.numChannels], 0, 0)

        if channel > 0:
            self.opacitySliders.append(QSlider(Qt.Vertical, self.widget))
            self.opacitySliders[self.numChannels].setRange(0, 255)
            self.opacitySliders[self.numChannels].setMaximumHeight(120)
            self.opacitySliders[self.numChannels].setMinimumWidth(20)

            self.opacitySliders[self.numChannels].setValue(int(0.85*255))
            thisMenuLayout.addWidget(self.opacitySliders[self.numChannels], 0, 1)
            self.opacitySliders[self.numChannels].valueChanged.connect(
                lambda value: self.adjustOpacity(value, channel))
        else:
            self.opacitySliders.append(0)

        self.gridMenu.addWidget(thisMenu, 0, self.numChannels+1, alignment=Qt.AlignTop)
        thisMenu.hide()

    def showMenu(self):
        """ Show menu when the button is pressed """
        if self.toggle == 0:
            # self.saturationSliders[0].setMinimumHeight(300)
            for frame in self.qFrames:
                frame.setVisible(True)

            self.toggle = 1
            self.toggleMenu.setMinimumHeight(180)
            self.toggleMenu.setMinimumWidth(130*self.numChannels+1)
        else:
            for frame in self.qFrames:
                frame.setVisible(False)
            self.toggle = 0
            self.toggleMenu.setFixedSize(40, 40)

    @pyqtSlot(int)
    def adjustOpacity(self, *values):
        """ React to the opacity slider and set the new value to the channels imageItem
        values will also get the value of the slider directly in values[0] """
        channel = values[1]
        opacity = self.opacitySliders[channel].value()/255
        self.imageItems[channel]['ImageItem'].setOpacity(opacity)

    @pyqtSlot(int)
    def adjustLevel(self, channel):
        """ React to a change in the saturation slider region and setLevels accordingly"""
        self.imageItems[channel]['ImageItem'].setLevels(
            self.saturationSliders[channel].regions[0].getRegion())

    @pyqtSlot(int)
    def updateImage(self, channel):
        """ Set new LUT to the channel ImageItem """
        self.imageItems[channel]['ImageItem'].setLookupTable(
            self.saturationSliders[channel].getLookupTable())


class GradientEditorWidget(pg.GraphicsView):
    """ Get GradientEditorItem into a Widget that can be added to GridBoxLayouts """
    def __init__(self, *args, **kargs):
        background = kargs.pop('background', 'default')
        pg.GraphicsView.__init__(self, *args, useOpenGL=False, background=background)
        self.item = pg.GradientEditorItem(*args, **kargs)
        self.setCentralItem(self.item)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setMaximumWidth(15)
        self.setMaximumHeight(150)

    def sizeHint(self):
        """ size hint to what this widget would like to be """
        return QtCore.QSize(115, 200)

    def __getattr__(self, attr):
        return getattr(self.item, attr)


class LUTItemSimple(QWidget):
    """ Simple bar that is colored like the underlying LUT. A right click opens the context
    Menu. Custom LUTs can be added to self.customGradients dict. Based on pg.HistogramLutItem, but
    much less computation intensive for QtImageViewerMerge. """
    gradientChanged = pyqtSignal([], [int])
    levelChanged = pyqtSignal([], [int])

    def __init__(self):
        QWidget.__init__(self)
        frame = QFrame(self)
        grid = QGridLayout(frame)
        self.setMinimumWidth(100)
        self.setMinimumHeight(150)

        self.gradient = GradientEditorWidget(parent=frame)
        self.gradient.setOrientation('right')
        grid.addWidget(self.gradient, 0, 0)
        self.gradient.sigGradientChangeFinished.connect(self.gradientChange)
        self.gradient.loadPreset('inferno')

        self.customGradients = pg.OrderedDict([('reds', {'ticks': [(0.0, (0, 0, 0, 255)),
                                                                   (1.0, (255, 0, 0, 255))],
                                                'mode': 'hsv'})])
        self.setCustomGradients()

        self.glw = pg.GraphicsLayoutWidget(parent=frame)

        grid.addWidget(self.glw, 0, 1)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.viewBox = self.glw.addViewBox()
        self.viewBox.setZValue(100)
        self.viewBox.setMaximumHeight(150)
        self.viewBox.setMaximumWidth(30)
        self.viewBox.setFlag(self.gradient.ItemStacksBehindParent)
        self.viewBox.sigRangeChanged.connect(self.update)
        self.viewBox.setMouseEnabled(x=False, y=True)
        self.viewBox.setYRange(0, 255)
        self.regions = [
            pg.LinearRegionItem([0, 255], 'horizontal', swapMode='push')]
        for region in self.regions:
            region.setZValue(1000)
            self.viewBox.addItem(region)
            region.lines[0].addMarker('<|', 0.5)
            region.lines[1].addMarker('|>', 0.5)
            region.sigRegionChanged.connect(self.regionChanging)
            region.sigRegionChangeFinished.connect(self.regionChanged)
        # self.regions[0].setSpan(0.8, 0.8)

        self.lut = None
        self.levelMode = 'mono'

    def customLutClicked(self):
        """ Pass the click on a custom LUT to self.gradient with the gradient instead of name """
        act = self.sender()
        self.gradient.restoreState(self.customGradients[act.name])

    def gradientChange(self):
        """ Emit if the gradient changed and delete the ticks """
        self.hideTicks()
        self.gradientChanged.emit()

    def hideTicks(self):
        """ Hide the ticks that are normally next to the LUT to adjust the LUT 'live' """
        ticks = self.gradient.listTicks()
        for tick in ticks:
            tick[0].hide()

    def getLookupTable(self, img=None, numColors=None, alpha=None):
        """Return a lookup table from the color gradient defined by this
        HistogramLUTItem.
        """
        if self.levelMode != 'mono':
            return None
        if numColors is None:
            if img is None:
                numColors = 256
            elif img.dtype == np.uint8:
                numColors = 256
            else:
                numColors = 512
        # if self.lut is None:
        self.lut = self.gradient.getLookupTable(numColors, alpha=alpha)
        return self.lut

    def regionChanging(self):
        """ Emit id the saturation slider is moving """
        self.levelChanged.emit()

    def regionChanged(self):
        """ Emit if the saturation slider has changes """
        self.levelChanged.emit()

    def loadPresetLUT(self, name):
        """ Fork to restoreState/loadPreset of self.gradient by passing the right variables """
        isSet = False
        for gradient in self.customGradients:
            if gradient == name:
                self.gradient.restoreState(self.customGradients[gradient])
                isSet = True
        if not isSet:
            self.gradient.loadPreset(name)

    def setCustomGradients(self):
        """ add the custom gradients defined in __init__ to the context menu """
        for gradient in self.customGradients:
            pixmap = QtGui.QPixmap(100, 15)
            painter = QtGui.QPainter(pixmap)
            self.gradient.restoreState(self.customGradients[gradient])
            grad = self.gradient.getGradient()
            brush = QtGui.QBrush(grad)
            painter.fillRect(QtCore.QRect(0, 0, 100, 15), brush)
            painter.end()
            label = QLabel()
            label.setPixmap(pixmap)
            label.setContentsMargins(1, 1, 1, 1)
            labelName = QLabel(gradient)
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(labelName)
            hbox.addWidget(label)
            widget = QtWidgets.QWidget()
            widget.setLayout(hbox)
            act = QtWidgets.QWidgetAction(self)
            act.setDefaultWidget(widget)
            act.triggered.connect(self.customLutClicked)
            act.name = gradient
            act.custom = True
            self.gradient.menu.insertAction(self.gradient.menu.actions()[-3], act)

# pg.ViewBox.viewRange.__get__


def main():
    """ Method to test the Viewer in a QBoxLayout with 1 and 2 channels"""

    fname = ('C:/Users/stepp/Documents/02_Raw/SmartMito/__short.tif')
    imageMitoOrig = io.imread(fname)
    imageDrpOrig = imageMitoOrig[1]
    imageMitoOrig = imageMitoOrig[0]
    mito = imageMitoOrig
    drp = imageDrpOrig

    app = QApplication(sys.argv)

    # win = pg.GraphicsScene()
    # viewBox = win.addViewBox()
# make sure this image is on top
    # img2.setOpacity(0.5)
    # pyqtgraph.ColorMap()
    # viewer.setColorMap(getPyQtcolormap())

    mode = 2

    if mode == 1:
        lutitem = LUTItemSimple()
        lutitem.show()
    elif mode == 2:
        win = pg.GraphicsWindow()

        # Show viewer and run application.
        viewer = QtImageViewerMerge()
        viewer2 = QtImageViewerMerge()

        grid = QGridLayout(win)
        grid.addWidget(viewer, 0, 0)
        grid.addWidget(viewer2, 0, 1)

        # imgItemdrp = viewer.addImage(drp)
        # imgItemmito = viewer.addImage(mito)
        # viewer.setLUT(imgItemdrp, 'reds')
        # viewer.setLUT(imgItemmito, 'grey')
        # viewer.setImage(drp, 0)
        # viewer.setImage(mito, 1)
        viewer.setImage(drp, 0)
        viewer.saturationSliders[0].viewBox.setYRange(0, 100)
        viewer.resetRanges()
        viewer2.addImage(mito)
        viewer2.resetRanges()
        viewer.cross.setPosition([(100, 100)])
        win.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
