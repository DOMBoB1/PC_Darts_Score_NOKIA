import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QMainWindow
from PySide6.QtCore import Qt, QTimer

class CameraFeedWidget(QWidget):
    def __init__(self, camera_number=1, parent=None, main_window=None):
        super().__init__(parent)
        self.camera_number = camera_number
        self.main_window = main_window  # Reference to main window for calibration params and checkbox
        self.setup_ui()
        self.setup_camera()
        self.setup_timer()

    def setup_ui(self):
        self.setLayout(QVBoxLayout())
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.image_label)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Make camera feed transparent to mouse events

    def setup_camera(self):
        self.capture = cv2.VideoCapture(self.camera_number)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open camera {self.camera_number}")

    def setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # Update every 20ms

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.timer.stop()
        self.capture.release()
        event.accept()

    def restart_camera(self):
        # Stop timer and release camera if open
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
            self.capture = None
        # Reopen camera
        self.setup_camera()
        self.setup_timer()
