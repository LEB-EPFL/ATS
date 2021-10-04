""" Fork of ImageViewerMerge that can be used to see a series and also have a slider to see all
frames. The slider looks better than QSlider and you can use the mouseWheel to scroll through the
frames if the mouse is located above the slider.
    """

import sys

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import QRectF, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout,
                             QInputDialog, QWidget)
from skimage import io

from SmartMicro.QtImageViewerMerge import QtImageViewerMerge


def main():
    """ Method to test the Viewer in a QBoxLayout with 1 and 2 channels"""
    app = QApplication(sys.argv)
    viewer = QtImageViewerSeries()
    fname = ('//lebnas1.epfl.ch/microsc125/Watchdog/Model/Mito.h5')
    # fname = ('W:/Watchdog/Model/Proc.h5')
    # fname = ('C:/Users/stepp/Documents/02_Raw/SmartMito/__short.tif')
    viewer.loadSeries(fname)
    viewer.viewer.cross.hide()
    viewer.viewer.crossRational.hide()
    # viewer.loadSeries(fname)
    # viewer.loadSeries(fname)
    viewer.show()
    sys.exit(app.exec_())


class QtImageViewerSeries(QWidget):
    """ Fork of ImageViewerMerge that can be used to see a series and also have a slider to see all
    """

    def __init__(self):
        QWidget.__init__(self)
        self.viewer = QtImageViewerMerge()
        self.grid = QGridLayout(self)
        # make a new slider that is much cooler
        self.slider = QNiceSlider()
        self.slider.sliderPressed.connect(self.startTimer)
        self.slider.sliderReleased.connect(self.stopTimer)
        self.slider.sliderChanged.connect(self.onTimer)

        self.grid.addWidget(self.viewer, 0, 0)
        self.grid.addWidget(self.slider, 1, 0)

        # get the timer ready for the slider
        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimer)
        self.timer.setInterval(20)

        self.series = None
        self.lastKey = None

    def newSlider(self, event):
        """ ??? """
        print(self.region.getRegion())
        print(event)

    def keyPressEvent(self, keyEvent: QtGui.QKeyEvent) -> None:
        """ Save the current frame that is displayed if Ctrl+Save is pressed """
        if self.lastKey == 16777249 and keyEvent.key() == 83:
            fname, *_ = QFileDialog.getSaveFileName(QWidget(), 'Save file')
            frame = int(self.slider.position)
            io.imsave(fname, self.series[frame, :, :])
        self.lastKey = keyEvent.key()
        return super().keyPressEvent(keyEvent)

    def onTimer(self, i=None):
        """Set the current frame if timer is running. Also accesible programmatically to set a
        certain frame by setting the i input
        """
        if i is None:
            i = int(self.slider.position)
        else:
            i = int(np.round(i))
        print(i)
        if self.series.ndim == 4:
            numChannels = self.series.shape[3]
        else:
            numChannels = 1

        if numChannels == 1:
            self.viewer.setImage(self.series[i, :, :], 0)
        else:
            for channel in range(numChannels):
                self.viewer.setImage(self.series[i, :, :, channel-1], channel-1)

    def loadSeries(self, filePath):
        """ Load anything that skimage imread can load as a series or a .h5 file """
        if filePath[-3:] == '.h5':
            fileHandle = h5py.File(filePath, 'r')
            if len(fileHandle.keys()) > 1:
                item, _ = QInputDialog.getItem(
                    self, "select Series", "series", fileHandle.keys(), 0, False)
            else:
                item = fileHandle.keys()[0]
            thisSeries = fileHandle.get(item)
            if self.series is None:
                self.series = np.array(thisSeries).astype('float')
            else:
                self.series = np.concatenate((self.series, np.array(thisSeries).astype('float')),
                                             axis=3)
            fileHandle.close()
            print(self.series.shape)
        else:
            if self.series is None:
                self.series = io.imread(filePath)
            else:
                self.series = np.concatenate((self.series, io.imread(filePath)), axis=2)
        if self.series.size == 4:
            self.viewer.addImage(self.series[0, :, :, self.series.shape[-1]-1])
        else:
            self.viewer.addImage(self.series[:, :, self.series.shape[-1]-1])
        self.viewer.rangeChanged()
        self.viewer.resetRanges()
        self.slider.setSliderRange([-1, self.series.shape[0]-1])
        self.viewer.resetZoom()

    def startTimer(self):
        """ start Timer when slider is pressed """
        self.timer.start()

    def stopTimer(self):
        """ stop timer when slider is released"""
        self.timer.stop()

    def nextFrame(self):
        """ display next frame in all viewBoxes when '>' button is pressed """
        i = self.frameSlider.value()
        self.frameSlider.setValue(i + 1)
        self.onTimer()

    def prevFrame(self):
        """ display previous frame in all viewBoxes when '<' button is pressed """
        i = self.frameSlider.value()
        self.frameSlider.setValue(i - 1)
        self.onTimer()


class QNiceSlider(pg.GraphicsLayoutWidget):
    """ This is a slider that looks much nicer than QSlider and also allows for mouse wheel
    scrolling. It has the same Signals as QSlider to allow for easy replacement"""
    sliderChanged = pyqtSignal(float)
    sliderPressed = pyqtSignal()
    sliderReleased = pyqtSignal()

    def __init__(self):
        pg.GraphicsLayoutWidget.__init__(self)
        self.setMaximumHeight(100)
        self.setMinimumHeight(100)
        self.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.viewB = self.addViewBox()
        self.viewB.setMouseEnabled(y=False, x=False)

        self.buttonWidth = 1
        self.position = 0
        self.span = [self.position-self.buttonWidth/2, self.position+self.buttonWidth/2]
        self.orientation = 'vertical'
        self.button = self.SliderButton(self.span, self.orientation)
        self.sliderBackground = self.SliderBackground()
        self.sliderBackground.sliderClicked.connect(self.sliderBgClicked)
        self.button.buttonChanged.connect(self.buttonChange)
        self.button.buttonPressed.connect(self.sliderPressed)
        self.button.buttonReleased.connect(self.sliderReleased)
        self.viewB.addItem(self.button)
        self.viewB.addItem(self.sliderBackground)
        self.viewB.setYRange(0, 1)
        self.sliderRange = [-1, 100]
        self.setSliderRange(self.sliderRange)

    def setSliderRange(self, sliderRange=None):
        """ Set the Range of the whole slider """
        if sliderRange is None:
            self.sliderRange = [-1, 100]
        else:
            self.sliderRange = sliderRange
        self.viewB.setXRange(sliderRange[0], sliderRange[1])
        self.button.setsliderRange(sliderRange)
        self.button.width = (sliderRange[1]-sliderRange[0])/100
        self.sliderBackground.setSliderBackground(sliderRange)

    def wheelEvent(self, ev):
        wheelCorrect = 120
        ev.accept()
        delta = ev.angleDelta().y()
        if self.sliderRange[1] > self.position + delta/wheelCorrect > self.sliderRange[0]:
            self.position = self.position + delta/wheelCorrect
        elif self.position + delta/wheelCorrect > self.sliderRange[1]:
            self.position = self.sliderRange[1]
        elif self.position + delta/wheelCorrect < self.sliderRange[0]:
            self.position = self.sliderRange[0]

        self.button.updatePosition(self.position)
        self.sliderChanged.emit(self.position)

    def buttonChange(self, pos):
        """  Notify if the Slider changed"""
        self.position = pos
        self.sliderChanged.emit(self.position)

    def sliderBgClicked(self, pos):
        """ Notify if the background of the slider was clicked to move the sliderButton there """
        self.position = pos
        self.sliderChanged.emit(self.position)
        self.button.updatePosition(self.position)

    class SliderBackground(pg.GraphicsObject):
        """ The Background of the slider. """
        sliderClicked = pyqtSignal(float)

        def __init__(self):
            pg.GraphicsObject.__init__(self)
            self.setZValue(500)
            self.currentBrush = QtGui.QBrush(QtGui.QColor(30, 30, 30, 200))
            self.railHeight = 0.6
            self.boundingR = QRectF(-1, (1-self.railHeight)/2, 100, self.railHeight)

        def paint(self, painter, *_):
            """ Reimplement the painter function """
            painter.setBrush(self.currentBrush)
            painter.setPen(pg.mkPen(None))
            painter.drawRect(self.boundingR)

        def setSliderBackground(self, rect):
            """ Set the background if sliderRange changed for example"""
            self.boundingR.setLeft(rect[0])
            self.boundingR.setRight(rect[1])
            self.prepareGeometryChange()

        def boundingRect(self):
            """ The bounding rectangle of the background """
            rectangle = self.boundingR
            rectangle = rectangle.normalized()
            return rectangle

        def mousePressEvent(self, event):
            """ Transmit where the background was clicked """
            self.sliderClicked.emit(event.pos().x())

    class SliderButton(pg.GraphicsObject):
        """ The button for the slider. """
        buttonChanged = pyqtSignal(float)
        buttonPressed = pyqtSignal()
        buttonReleased = pyqtSignal()

        def __init__(self, span, orientation):
            pg.GraphicsObject.__init__(self)
            self.setZValue(1000)
            self.currentBrush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 200))

            self.span = span
            self.width = span[1]-span[0]
            self.orientation = orientation
            self.sliderRange = [-1, 100]

        def setsliderRange(self, sliderRange):
            """ Set the internal sliderRange so the button knows where it is allowed to go"""
            self.sliderRange = sliderRange

        def boundingRect(self, span=None):
            """ The rectangle that is the button """
            rectangle = QRectF(self.viewRect())
            if span is None:
                rng = self.span
            else:
                rng = span

            if self.orientation == 'vertical':
                rectangle.setLeft(rng[0])
                rectangle.setRight(rng[1])
                length = rectangle.height()
                rectangle.setBottom(0)
                rectangle.setTop(1)
            else:
                rectangle.setTop(rng[0])
                rectangle.setBottom(rng[1])
                length = rectangle.width()
                rectangle.setRight(rectangle.left() + length * self.span[1])
                rectangle.setLeft(rectangle.left() + length * self.span[0])

            rectangle = rectangle.normalized()
            return rectangle

        def paint(self, painter, *_):
            """ Redefine the paint event """
            painter.setBrush(self.currentBrush)
            painter.setPen(pg.mkPen(None))
            painter.drawRect(self.boundingRect())

        def updatePosition(self, pos):
            """ Prepare the paint to know where to paint """
            self.span = [pos-self.width/2, pos+self.width/2]
            self.prepareGeometryChange()
            self.buttonChanged.emit(pos)

        def mouseMoveEvent(self, event):
            """ Make the button follow the mouse if in the correct range """
            if self.sliderRange[1] > event.lastPos().x() > self.sliderRange[0]:
                self.updatePosition(event.lastPos().x())

        def mousePressEvent(self, _):
            """ Transmit button press to the main slider class """
            self.buttonPressed.emit()

        def mouseReleaseEvent(self, _):
            """ Transmit button release to the main slider class """
            self.buttonReleased.emit()


if __name__ == '__main__':
    main()
