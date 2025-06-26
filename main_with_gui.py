import os
import pickle
import sys
from statistics import mode
from time import sleep

import cv2
import numpy as np
from camera_feed import CameraFeedWidget
from PySide6 import QtCore 
from PySide6.QtCore import QThreadPool, QRunnable, Signal, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGroupBox, QGridLayout

# Camera configuration
CAMERA_NUMBER = 1  # Change this to your camera index

import CalibrationWithUncertainty
import ContourUtils
from Dart_Scoring import dart_scorer_util, DartScore
import opencv_gui_sliders
import utils
from FPS import FPS
from QT_GUI_Elements.qt_ui_classes import DartPositionLabel
from QT_GUI_Elements.ui_dart_main_gui import Ui_DartScorer

# #############  Config  ####################
USE_CAMERA_CALIBRATION_TO_UNDISTORT = True
loadSavedParameters = True
CAMERA_NUMBER = 1  # 0,1 is built-in, 2 is external webcam
TRIANGLE_DETECT_THRESH = 11
minArea = 800
maxArea = 4000
score1 = DartScore.Score(501, True)
score2 = DartScore.Score(501, True)
scored_values = []
scored_mults = []

# Globals
points = []
dart_tip = None
ACTIVE_PLAYER = 1
UNDO_LAST_FLAG = False
DARTBOARD_AREA = 0
center_ellipse = (0, 0)
values_of_round = []
mults_of_round = []
current_settings = None
OPENCV_GUI_CREATED = False

new_dart_tip = None
update_dart_point = False

ellipse = None
x_offset_current, y_offset_current = 0, 0

#############################################
STOP_DETECTION = False
dart_id = 0

# Initialize camera in background
thread_pool = QThreadPool()
print(f"Multithreading with maximum {thread_pool.maxThreadCount()} threads")

# Store initialized camera and calibration data - these will be set in MainWindow
target_ROI_size = (600, 600)
resize_for_squish = (600, 600)  # Squish the image if the circle doesnt quite fit
dart_board_in_gui_dimensions = (501, 501)
Scaling_factor_for_x_placing_in_gui = (dart_board_in_gui_dimensions[0] / resize_for_squish[0], dart_board_in_gui_dimensions[1] / resize_for_squish[1])

previous_img = np.zeros((target_ROI_size[0], target_ROI_size[1], 3)).astype(np.uint8)
difference = np.zeros(target_ROI_size).astype(np.uint8)
img_undist = np.zeros(target_ROI_size).astype(np.uint8)

default_img = None


class ManualCalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Camera Calibration")
        self.setModal(True)
        self.resize(800, 600)
        
        # Initialize calibration parameters
        self.fx = 1000.0  # Focal length x
        self.fy = 1000.0  # Focal length y
        self.cx = 960.0   # Principal point x (half of 1920)
        self.cy = 540.0   # Principal point y (half of 1080)
        self.k1 = 0.0     # Radial distortion coefficient 1
        self.k2 = 0.0     # Radial distortion coefficient 2
        self.p1 = 0.0     # Tangential distortion coefficient 1
        self.p2 = 0.0     # Tangential distortion coefficient 2
        
        # Get camera instance from parent
        self.camera_instance = parent
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout()
        
        # Left side - Controls
        controls_layout = QVBoxLayout()
        
        # Focal length controls
        focal_group = QGroupBox("Focal Length")
        focal_layout = QGridLayout()
        
        self.fx_slider = QSlider()
        self.fx_slider.setRange(500, 2000)
        self.fx_slider.setValue(int(self.fx))
        self.fx_slider.valueChanged.connect(self.update_fx)
        self.fx_label = QLabel(f"fx: {self.fx:.1f}")
        
        self.fy_slider = QSlider()
        self.fy_slider.setRange(500, 2000)
        self.fy_slider.setValue(int(self.fy))
        self.fy_slider.valueChanged.connect(self.update_fy)
        self.fy_label = QLabel(f"fy: {self.fy:.1f}")
        
        focal_layout.addWidget(QLabel("Focal Length X:"), 0, 0)
        focal_layout.addWidget(self.fx_slider, 0, 1)
        focal_layout.addWidget(self.fx_label, 0, 2)
        focal_layout.addWidget(QLabel("Focal Length Y:"), 1, 0)
        focal_layout.addWidget(self.fy_slider, 1, 1)
        focal_layout.addWidget(self.fy_label, 1, 2)
        focal_group.setLayout(focal_layout)
        
        # Principal point controls
        principal_group = QGroupBox("Principal Point")
        principal_layout = QGridLayout()
        
        self.cx_slider = QSlider()
        self.cx_slider.setRange(800, 1120)
        self.cx_slider.setValue(int(self.cx))
        self.cx_slider.valueChanged.connect(self.update_cx)
        self.cx_label = QLabel(f"cx: {self.cx:.1f}")
        
        self.cy_slider = QSlider()
        self.cy_slider.setRange(400, 680)
        self.cy_slider.setValue(int(self.cy))
        self.cy_slider.valueChanged.connect(self.update_cy)
        self.cy_label = QLabel(f"cy: {self.cy:.1f}")
        
        principal_layout.addWidget(QLabel("Principal Point X:"), 0, 0)
        principal_layout.addWidget(self.cx_slider, 0, 1)
        principal_layout.addWidget(self.cx_label, 0, 2)
        principal_layout.addWidget(QLabel("Principal Point Y:"), 1, 0)
        principal_layout.addWidget(self.cy_slider, 1, 1)
        principal_layout.addWidget(self.cy_label, 1, 2)
        principal_group.setLayout(principal_layout)
        
        # Distortion controls
        distortion_group = QGroupBox("Distortion Coefficients")
        distortion_layout = QGridLayout()
        
        self.k1_slider = QSlider()
        self.k1_slider.setRange(-100, 100)
        self.k1_slider.setValue(int(self.k1 * 100))
        self.k1_slider.valueChanged.connect(self.update_k1)
        self.k1_label = QLabel(f"k1: {self.k1:.3f}")
        
        self.k2_slider = QSlider()
        self.k2_slider.setRange(-100, 100)
        self.k2_slider.setValue(int(self.k2 * 100))
        self.k2_slider.valueChanged.connect(self.update_k2)
        self.k2_label = QLabel(f"k2: {self.k2:.3f}")
        
        self.p1_slider = QSlider()
        self.p1_slider.setRange(-50, 50)
        self.p1_slider.setValue(int(self.p1 * 100))
        self.p1_slider.valueChanged.connect(self.update_p1)
        self.p1_label = QLabel(f"p1: {self.p1:.3f}")
        
        self.p2_slider = QSlider()
        self.p2_slider.setRange(-50, 50)
        self.p2_slider.setValue(int(self.p2 * 100))
        self.p2_slider.valueChanged.connect(self.update_p2)
        self.p2_label = QLabel(f"p2: {self.p2:.3f}")
        
        distortion_layout.addWidget(QLabel("Radial Distortion k1:"), 0, 0)
        distortion_layout.addWidget(self.k1_slider, 0, 1)
        distortion_layout.addWidget(self.k1_label, 0, 2)
        distortion_layout.addWidget(QLabel("Radial Distortion k2:"), 1, 0)
        distortion_layout.addWidget(self.k2_slider, 1, 1)
        distortion_layout.addWidget(self.k2_label, 1, 2)
        distortion_layout.addWidget(QLabel("Tangential Distortion p1:"), 2, 0)
        distortion_layout.addWidget(self.p1_slider, 2, 1)
        distortion_layout.addWidget(self.p1_label, 2, 2)
        distortion_layout.addWidget(QLabel("Tangential Distortion p2:"), 3, 0)
        distortion_layout.addWidget(self.p2_slider, 3, 1)
        distortion_layout.addWidget(self.p2_label, 3, 2)
        distortion_group.setLayout(distortion_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Default")
        self.reset_button.clicked.connect(self.reset_to_default)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        
        # Add all components to controls layout
        controls_layout.addWidget(focal_group)
        controls_layout.addWidget(principal_group)
        controls_layout.addWidget(distortion_group)
        controls_layout.addLayout(button_layout)
        controls_layout.addStretch()
        
        # Right side - Preview
        preview_group = QGroupBox("Calibration Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel("Camera Preview")
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        
        # Add both sides to main layout
        layout.addLayout(controls_layout)
        layout.addWidget(preview_group)
        
        self.setLayout(layout)
        
        # Start preview timer
        self.preview_timer = QtCore.QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(100)  # Update every 100ms
    
    def update_preview(self):
        """Update the calibration preview"""
        try:
            if self.camera_instance and self.camera_instance.cap and self.camera_instance.cap.isOpened():
                success, frame = self.camera_instance.cap.read()
                if success and frame is not None:
                    # Apply current calibration parameters
                    camera_matrix, dist_coeffs = self.get_calibration_matrices()
                    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
                    
                    # Resize for display
                    height, width = undistorted.shape[:2]
                    display_width = 400
                    display_height = int(height * display_width / width)
                    display_frame = cv2.resize(undistorted, (display_width, display_height))
                    
                    # Convert to Qt format
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    
                    self.preview_label.setPixmap(pixmap)
                else:
                    self.preview_label.setText("Camera not responding")
            else:
                self.preview_label.setText("Camera not available")
        except Exception as e:
            self.preview_label.setText(f"Preview error: {str(e)[:50]}")
            print(f"Preview error: {e}")
    
    def update_fx(self, value):
        self.fx = float(value)
        self.fx_label.setText(f"fx: {self.fx:.1f}")
        
    def update_fy(self, value):
        self.fy = float(value)
        self.fy_label.setText(f"fy: {self.fy:.1f}")
        
    def update_cx(self, value):
        self.cx = float(value)
        self.cx_label.setText(f"cx: {self.cx:.1f}")
        
    def update_cy(self, value):
        self.cy = float(value)
        self.cy_label.setText(f"cy: {self.cy:.1f}")
        
    def update_k1(self, value):
        self.k1 = value / 100.0
        self.k1_label.setText(f"k1: {self.k1:.3f}")
        
    def update_k2(self, value):
        self.k2 = value / 100.0
        self.k2_label.setText(f"k2: {self.k2:.3f}")
        
    def update_p1(self, value):
        self.p1 = value / 100.0
        self.p1_label.setText(f"p1: {self.p1:.3f}")
        
    def update_p2(self, value):
        self.p2 = value / 100.0
        self.p2_label.setText(f"p2: {self.p2:.3f}")
        
    def reset_to_default(self):
        self.fx = 1000.0
        self.fy = 1000.0
        self.cx = 960.0
        self.cy = 540.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0
        
        self.fx_slider.setValue(int(self.fx))
        self.fy_slider.setValue(int(self.fy))
        self.cx_slider.setValue(int(self.cx))
        self.cy_slider.setValue(int(self.cy))
        self.k1_slider.setValue(int(self.k1 * 100))
        self.k2_slider.setValue(int(self.k2 * 100))
        self.p1_slider.setValue(int(self.p1 * 100))
        self.p2_slider.setValue(int(self.p2 * 100))
    
    def closeEvent(self, event):
        """Stop the preview timer when dialog is closed"""
        self.preview_timer.stop()
        super().closeEvent(event)
    
    def get_calibration_matrices(self):
        """Return the camera matrix and distortion coefficients"""
        camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.array([self.k1, self.k2, self.p1, self.p2, 0.0], dtype=np.float64)
        
        return camera_matrix, dist_coeffs


class UIFunctions:

    def update_detection_sensitivity(self):
        global TRIANGLE_DETECT_THRESH
        TRIANGLE_DETECT_THRESH = self.ui.detection_sensitivity_slider.value()
        self.ui.current_detection_sensitivity_lable.setText(f"{TRIANGLE_DETECT_THRESH}")

    def undo_last_throw(self):
        global values_of_round, mults_of_round, UNDO_LAST_FLAG
        UNDO_LAST_FLAG = True
        self.ui.press_enter_label.setText("    Please throw again")
        if len(values_of_round) > 0:
            val = values_of_round.pop()
            mult = mults_of_round.pop()
            if ACTIVE_PLAYER == 1:
                current_sum = int(self.ui.player1_sum_round.text()) if self.ui.player1_sum_round.text() else 0
                self.ui.player1_sum_round.setText(str(current_sum - val * mult))
                if len(values_of_round) == 0:
                    self.ui.player1_1.setText("")
                elif len(values_of_round) == 1:
                    self.ui.player1_2.setText("")
                elif len(values_of_round) == 2:
                    self.ui.player1_3.setText("")
            elif ACTIVE_PLAYER == 2:
                current_sum = int(self.ui.player2_sum_round.text()) if self.ui.player2_sum_round.text() else 0
                self.ui.player2_sum_round.setText(str(current_sum - val * mult))
                if len(values_of_round) == 0:
                    self.ui.player2_1.setText("")
                elif len(values_of_round) == 1:
                    self.ui.player2_2.setText("")
                elif len(values_of_round) == 2:
                    self.ui.player2_3.setText("")

            # Remove the last dart position from the board
            if self.DartPositions:
                last_key = list(self.DartPositions.keys())[-1]
                self.DartPositions[last_key].setText("")
                del self.DartPositions[last_key]

    def delete_all_x_on_board(self):
        print("LEN:", len(self.DartPositions.values()))
        for label in self.DartPositions.values():
            label.setText("")
        self.DartPositions = {}

    def place_x_on_board(self, pos_x, pos_y):
        global dart_id
        DartPositionId = dart_id
        dart_id = dart_id + 1
        self.DartPositions[DartPositionId] = DartPositionLabel(self.ui.dart_board_image)
        print(f"Placing: {str(int(pos_x*Scaling_factor_for_x_placing_in_gui[0])), str(int(pos_y*Scaling_factor_for_x_placing_in_gui[1]))}")
        self.DartPositions[DartPositionId].addDartPosition(int(pos_x * Scaling_factor_for_x_placing_in_gui[0]),
                                                           int(pos_y * Scaling_factor_for_x_placing_in_gui[1]))

    def set_default_image(self):
        pool = QThreadPool.globalInstance()
        default_img_setter = DefaultImageSetter(self)
        pool.start(default_img_setter)

    def start_detection_and_scoring(self):
        global STOP_DETECTION, default_img, img_undist
        STOP_DETECTION = False
        # change color of stop_measuring_button to transparent
        self.ui.stop_measuring_button.setStyleSheet("background-color: transparent")
        # change color of start_measuring_button to green
        self.ui.start_measuring_button.setStyleSheet("background-color: green")
        self.ui.press_enter_label.setText("")
        pool = QThreadPool.globalInstance()
        default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
        detection_and_scoring = DetectionAndScoring(self)
        self.delete_all_x_on_board()
        pool.start(detection_and_scoring)

    def stop_detection_and_scoring(self):
        global STOP_DETECTION
        # change color of stop_measuring_button to red
        self.ui.stop_measuring_button.setStyleSheet("background-color: red")
        self.ui.start_measuring_button.setStyleSheet("background-color: transparent")
        self.ui.press_enter_label.setText("    1. Remove all Darts\n    2. Press Continue to start next round")
        STOP_DETECTION = True

    def update_game_settings(self):
        score = int(self.ui.initial_score_comboBox.currentText())
        score1.setNominalScore(score)
        score2.setNominalScore(score)

    def update_labels(self):
        global values_of_round, mults_of_round, ACTIVE_PLAYER, new_dart_tip, update_dart_point
        if update_dart_point and new_dart_tip is not None:
            print(f"Updating dart point in image {new_dart_tip[0], new_dart_tip[1]}")
            X_OFFSET = 8
            Y_OFFSET = 17
            self.place_x_on_board(new_dart_tip[0] -X_OFFSET, new_dart_tip[1]- Y_OFFSET)

            update_dart_point = False
        if ACTIVE_PLAYER == 1:
            self.ui.player_frame.setStyleSheet("background-color: #3a3a3a;")
            self.ui.player_frame_2.setStyleSheet("background-color: rgb(35, 35, 35);")
            # self.ui.player1_sum_round.setText(str(sum(values_of_round)))
            if len(values_of_round) == 1:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0] * mults_of_round[0]))
            elif len(values_of_round) == 2:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0] * mults_of_round[0] + values_of_round[1] * mults_of_round[1]))
            elif len(values_of_round) == 3:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player1_3.setText(f"{values_of_round[2] * mults_of_round[2]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0] * mults_of_round[0] + values_of_round[1] * mults_of_round[1] + values_of_round[2] * mults_of_round[2]))
            else:
                self.ui.player1_1.setText("-")
                self.ui.player1_2.setText("-")
                self.ui.player1_3.setText("-")
                self.ui.player1_sum_round.setText("")
        elif ACTIVE_PLAYER == 2:
            self.ui.player_frame_2.setStyleSheet("background-color: #3a3a3a;")
            self.ui.player_frame.setStyleSheet("background-color: rgb(35, 35, 35);")
            if len(values_of_round) == 1:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0] * mults_of_round[0]))
            elif len(values_of_round) == 2:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0] * mults_of_round[0] + values_of_round[1] * mults_of_round[1]))
            elif len(values_of_round) == 3:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player2_3.setText(f"{values_of_round[2] * mults_of_round[2]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0] * mults_of_round[0] + values_of_round[1] * mults_of_round[1] + values_of_round[2] * mults_of_round[2]))
            else:
                self.ui.player2_1.setText("-")
                self.ui.player2_2.setText("-")
                self.ui.player2_3.setText("-")
                self.ui.player2_sum_round.setText("")

        # if one of the players has won the game, show the winner
        if score1.currentScore == 0:
            self.warning("Player 1 has won the game!")
        elif score2.currentScore == 0:
            self.warning("Player 2 has won the game!")

        self.ui.player1_overall.setText(str(score1.currentScore))
        self.ui.player2_overall.setText(str(score2.currentScore))

    def test_camera(self):
        """Test camera functionality and provide diagnostic information"""
        try:
            if not self.cap or not self.cap.isOpened():
                QMessageBox.warning(self, "Camera Test", "Camera is not available or not opened.")
                return
            
            # Test camera reading
            ret, frame = self.cap.read()
            if not ret or frame is None:
                QMessageBox.warning(self, "Camera Test", "Camera is not responding. Cannot read frames.")
                return
            
            # Get camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Show test image
            test_window = QDialog(self)
            test_window.setWindowTitle("Camera Test - Press any key to close")
            test_window.resize(640, 480)
            
            layout = QVBoxLayout()
            
            # Camera info
            info_label = QLabel(f"Camera Info:\nResolution: {width}x{height}\nFPS: {fps:.1f}\nFrame size: {frame.shape}")
            layout.addWidget(info_label)
            
            # Image display
            image_label = QLabel()
            image_label.setMinimumSize(400, 300)
            
            # Convert frame to Qt format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            
            layout.addWidget(image_label)
            
            # Status
            status_label = QLabel("Camera is working correctly!")
            status_label.setStyleSheet("color: green; font-weight: bold;")
            layout.addWidget(status_label)
            
            test_window.setLayout(layout)
            test_window.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Camera Test Error", f"Error testing camera: {e}")


class MainWindow(QMainWindow, UIFunctions):
    def __init__(self):
        super().__init__()
        self.ui = Ui_DartScorer()
        self.ui.setupUi(self)  # Set up the external generated ui
        self.setWindowIcon(QIcon('icons/dart_icon.ico'))
        self.setWindowTitle("Dart Master")

        # Initialize camera and calibration data
        self.cap = None
        self.meanMTX = None
        self.meanDIST = None
        self.initialize_camera()

        # Create camera feed widget and place it in the layout
        # Create camera feed widget with optimized position
        self.camera_feed = CameraFeedWidget()
        self.camera_feed.setGeometry(540, 30, 480, 600)  # Moved further to the right
        self.camera_feed.setParent(self)
        self.camera_feed.show()

        # Add a border to the camera feed for better visibility
        self.camera_feed.setStyleSheet("""
            border: 0px solid #b78620;
            border-radius: 0px;
            background-color: transparent;
        """)

        # Add Calibration Button
        self.calibrate_button = QPushButton("Calibrate Camera", self)
        self.calibrate_button.setGeometry(690, 30, 100, 30)  # Position the button
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.calibrate_button.show()
        
        # Add Camera Test Button
        self.test_camera_button = QPushButton("Test Camera", self)
        self.test_camera_button.setGeometry(690, 70, 100, 30)  # Position below calibrate button
        self.test_camera_button.clicked.connect(self.test_camera)
        self.test_camera_button.show()

        # Buttons - Fixed connections
        self.ui.set_default_img_button.clicked.connect(self.set_default_image)
        self.ui.start_measuring_button.clicked.connect(self.start_detection_and_scoring)
        self.ui.stop_measuring_button.clicked.connect(self.stop_detection_and_scoring)
        self.ui.undo_last_throw_button.clicked.connect(self.undo_last_throw)
        self.ui.initial_score_comboBox.currentIndexChanged.connect(self.update_game_settings)
        self.ui.detection_sensitivity_slider.valueChanged.connect(self.update_detection_sensitivity)
        self.ui.continue_button.clicked.connect(self.start_detection_and_scoring)
        self.DartPositions = {}
        template_btn = self.ui.undo_last_throw_button
        self.calibrate_button.setFont(template_btn.font())
        self.calibrate_button.setPalette(template_btn.palette()) 
        self.calibrate_button.setSizePolicy(template_btn.sizePolicy())  
        self.calibrate_button.setMinimumSize(template_btn.minimumSize())
        self.calibrate_button.setStyleSheet(template_btn.styleSheet())       

        self.show()

    def initialize_camera(self):
        """Initialize camera and load calibration data"""
        try:
            # Try to find a working camera
            self.cap = self.find_working_camera()
            if not self.cap or not self.cap.isOpened():
                raise ValueError("Could not find a working camera")

            # Try different camera settings if the default ones fail
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Test if camera is working
            ret, test_frame = self.cap.read()
            if not ret:
                # Try lower resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, test_frame = self.cap.read()
                if not ret:
                    raise ValueError("Camera is not responding even with lower resolution")

            if USE_CAMERA_CALIBRATION_TO_UNDISTORT:
                if loadSavedParameters:
                    try:
                        pickle_in_MTX = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PickleFiles", "mtx_cheap_webcam_good_target.pickle"), "rb")
                        self.meanMTX = pickle.load(pickle_in_MTX)
                        print(self.meanMTX)

                        pickle_in_DIST = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PickleFiles", "dist_cheap_webcam_good_target.pickle"), "rb")
                        self.meanDIST = pickle.load(pickle_in_DIST)
                        print(self.meanDIST)
                        print("Parameters Loaded")
                    except FileNotFoundError:
                        print("Calibration files not found. Starting calibration...")
                        self.perform_calibration_with_manual_fallback()
                else:
                    self.perform_calibration_with_manual_fallback()
            print("Camera initialization completed")
        except Exception as e:
            print(f"Error during camera initialization: {e}")
            # Set default calibration parameters if camera fails
            self.set_default_calibration_parameters()
            QMessageBox.warning(self, "Camera Warning", 
                              f"Camera initialization failed: {e}\n\nUsing default calibration parameters.\nYou can recalibrate manually later.")

    def find_working_camera(self):
        """Try different camera indices to find a working camera"""
        camera_indices = [CAMERA_NUMBER, 0, 1, 2, 3]  # Try default first, then common indices
        
        for idx in camera_indices:
            try:
                print(f"Trying camera index {idx}...")
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Found working camera at index {idx}")
                        return cap
                    else:
                        cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"Error with camera index {idx}: {e}")
                continue
        
        print("No working camera found")
        return None

    def set_default_calibration_parameters(self):
        """Set default calibration parameters when camera fails"""
        self.meanMTX = np.array([
            [1000.0, 0, 960.0],
            [0, 1000.0, 540.0],
            [0, 0, 1]
        ], dtype=np.float64)
        self.meanDIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        print("Default calibration parameters set")

    def perform_calibration_with_manual_fallback(self):
        """Perform calibration with automatic fallback to manual calibration"""
        try:
            print("Attempting automatic calibration...")
            # Check if camera is working
            if not self.cap or not self.cap.isOpened():
                raise ValueError("Camera not available for calibration")
                
            self.meanMTX, self.meanDIST, _, _ = CalibrationWithUncertainty.calibrateCamera(
                cap=self.cap, rows=6, columns=9, squareSize=30, runs=1, saveImages=False, webcam=True
            )
            print("Automatic calibration successful!")
            self.save_calibration_parameters()
        except Exception as e:
            print(f"Automatic calibration failed: {e}")
            print("Falling back to manual calibration...")
            self.perform_manual_calibration()

    def perform_manual_calibration(self):
        """Perform manual calibration using the dialog"""
        try:
            dialog = ManualCalibrationDialog(self)
            if dialog.exec() == QDialog.Accepted:
                self.meanMTX, self.meanDIST = dialog.get_calibration_matrices()
                print("Manual calibration applied!")
                print(f"Camera Matrix:\n{self.meanMTX}")
                print(f"Distortion Coefficients:\n{self.meanDIST}")
                self.save_calibration_parameters()
            else:
                print("Manual calibration cancelled. Using default parameters.")
                self.set_default_calibration_parameters()
        except Exception as e:
            print(f"Manual calibration failed: {e}")
            self.set_default_calibration_parameters()

    def save_calibration_parameters(self):
        """Save calibration parameters to pickle files"""
        try:
            pickle_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PickleFiles")
            os.makedirs(pickle_dir, exist_ok=True)
            
            # Save camera matrix
            with open(os.path.join(pickle_dir, "mtx_cheap_webcam_good_target.pickle"), "wb") as f:
                pickle.dump(self.meanMTX, f)
            
            # Save distortion coefficients
            with open(os.path.join(pickle_dir, "dist_cheap_webcam_good_target.pickle"), "wb") as f:
                pickle.dump(self.meanDIST, f)
            
            print("Calibration parameters saved successfully!")
        except Exception as e:
            print(f"Failed to save calibration parameters: {e}")

    def start_calibration(self):
        print("Starting camera calibration...")
        try:
            # Check if camera is available
            if not self.cap or not self.cap.isOpened():
                QMessageBox.warning(self, "Camera Not Available", 
                                  "Camera is not available. Please check your camera connection and try again.")
                return
            
            # Ask user if they want automatic or manual calibration
            reply = QMessageBox.question(
                self, 
                "Calibration Method", 
                "Choose calibration method:\n\nYes = Automatic calibration (requires calibration pattern)\nNo = Manual calibration (adjust parameters manually)\nCancel = Skip calibration",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                # Automatic calibration
                try:
                    rows = 6
                    columns = 9
                    square_size = 30  # Size of the calibration squares
                    runs = 1  # Number of runs for calibration
                    self.meanMTX, self.meanDIST, uncertaintyMTX, uncertaintyDIST = CalibrationWithUncertainty.calibrateCamera(
                        cap=self.cap, rows=rows, columns=columns, squareSize=square_size, runs=runs
                    )
                    print("Automatic calibration complete. Camera matrix and distortion coefficients saved.")
                    self.save_calibration_parameters()
                    QMessageBox.information(self, "Calibration Complete", "Automatic camera calibration completed successfully!")
                except Exception as e:
                    print(f"Automatic calibration error: {e}")
                    reply2 = QMessageBox.question(self, "Calibration Failed", 
                                                f"Automatic calibration failed: {e}\n\nWould you like to try manual calibration?",
                                                QMessageBox.Yes | QMessageBox.No)
                    if reply2 == QMessageBox.Yes:
                        self.perform_manual_calibration()
                    else:
                        self.set_default_calibration_parameters()
                        
            elif reply == QMessageBox.No:
                # Manual calibration
                self.perform_manual_calibration()
                QMessageBox.information(self, "Calibration Complete", "Manual camera calibration completed!")
            else:
                print("Calibration cancelled by user.")
                self.set_default_calibration_parameters()
                
        except Exception as e:
            print(f"Calibration error: {e}")
            QMessageBox.critical(self, "Calibration Error", f"Calibration failed: {e}")
            self.set_default_calibration_parameters()

    def warning(self, message="Default"):
        QMessageBox.about(self, "Congratulations !", message)


class DefaultImageSetter(QRunnable):
    def __init__(self, camera_instance):
        super().__init__()
        self.camera_instance = camera_instance

    def run(self):
        global default_img, markerCorners, markerIds
        found_markers = False
        while True:
            success, img = self.camera_instance.cap.read()
            if success:
                if USE_CAMERA_CALIBRATION_TO_UNDISTORT:
                    img_undist = utils.undistortFunction(img, self.camera_instance.meanMTX, self.camera_instance.meanDIST)
                else:
                    img_undist = img
                img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=False, draw=True)
                if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
                    img_roi = cv2.resize(img_roi, resize_for_squish)
                    default_img = img_roi
                    print("Set default image")
                    cv2.imshow("Default", default_img)
                    cv2.waitKey(1)
                    found_markers = True
                if found_markers:
                    cv2.putText(img_undist, "Found markers press/hold x to save", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if cv2.waitKey(1) & 0xff == ord('x'):
                        cv2.destroyWindow("Preview")
                        cv2.destroyWindow("Default")
                        break
                else:
                    cv2.putText(img_undist, "No markers found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Preview", img_undist)


def get_biggest_contour(contours):
    """
    Get the biggest contour from a list of contours
    :param contours:
    :return:
    """
    contour = None
    for contour in contours:
        # if there are still multiple contours, take the one with the biggest area
        if len(contours) > 1:
            contour = max(contours, key=cv2.contourArea)
        # get the points of the contour
    return contour


class DetectionAndScoring(QRunnable):
    def __init__(self, camera_instance):
        global points, dart_tip, TRIANGLE_DETECT_THRESH, \
            score1, score2, scored_values, scored_mults, mults_of_round, values_of_round, img_undist, default_img, OPENCV_GUI_CREATED
        super().__init__()
        self.camera_instance = camera_instance
        if not OPENCV_GUI_CREATED:  # create the gui only once
            opencv_gui_sliders.create_gui()
            OPENCV_GUI_CREATED = True
        default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
        cv2.destroyWindow("Object measurement")

    def run(self):
        global previous_img, difference, default_img, ACTIVE_PLAYER, UNDO_LAST_FLAG
        global points, dart_tip, TRIANGLE_DETECT_THRESH, score1, score2, scored_values, scored_mults, mults_of_round, values_of_round
        global new_dart_tip, update_dart_point, minArea, DARTBOARD_AREA
        while True:
            if STOP_DETECTION:
                break
            fpsReader = FPS()
            success, img = self.camera_instance.cap.read()
            if success:
                if USE_CAMERA_CALIBRATION_TO_UNDISTORT:
                    img_undist = utils.undistortFunction(img, self.camera_instance.meanMTX, self.camera_instance.meanDIST)
                else:
                    img_undist = img
                img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=False, hold_position=True)
                if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
                    img_roi = cv2.resize(img_roi, resize_for_squish)
                    # resize img by a factor of 2
                    img_show = cv2.resize(img_roi, dsize=(400, 400))
                    cv2.imshow("Live", img_show)

                    # cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automaticMode, threshold_new = gui.updateTrackBar()

                    ret = detect_dart_circle_and_set_limits(img_roi=img_roi)
                    if center_ellipse == (0, 0):  # If dartboard was never detected raise exception
                        print("No dartboard detected!")
                        continue

                    # get the difference image
                    if default_img is None or np.all(default_img == 0):  # TODO: Bad fix but works
                        default_img = img_roi.copy()

                    difference = cv2.absdiff(img_roi, default_img)
                    # blur it for better edges
                    gray, thresh = self.prepare_differnce_image(TRIANGLE_DETECT_THRESH, difference)

                    minimal_darts_area = 0.005 * DARTBOARD_AREA  # Darts are > 0.5% of the dartboard area
                    maximal_darts_area = 0.1 * DARTBOARD_AREA  # Darts are < 10% of the dartboard area
                    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    noise_contours = [i for i in contours if cv2.contourArea(i) < minimal_darts_area]
                    darts_contours = [i for i in contours if minimal_darts_area < cv2.contourArea(i) < maximal_darts_area]  # Filter out contours that are too small or too big
                    if len(noise_contours) > 10 and len(darts_contours) == 0:
                        print("Too much noise")
                        default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
                        continue
                    # contour = get_biggest_contour(contours)  # Get the biggest contour
                    # if contour is None:
                    #     continue  # If no contour was found continue with next frame
                    for contour in darts_contours:
                        points_list = contour.reshape(contour.shape[0], contour.shape[2])
                        triangle = cv2.minEnclosingTriangle(cv2.UMat(points_list.astype(np.float32)))
                        triangle_np_array = cv2.UMat.get(triangle[1])
                        if triangle_np_array is not None:
                            pt1, pt2, pt3 = triangle_np_array.astype(np.int32)
                        else:
                            pt1, pt2, pt3 = np.array([-1, -1]), np.array([-1, -1]), np.array([-1, -1])

                        dart_tip, rest_pts = dart_scorer_util.find_tip_of_dart(pt1, pt2, pt3)
                        # Display the Dart point
                        cv2.circle(img_roi, dart_tip, 4, (0, 0, 255), -1)

                        self.draw_detected_darts(dart_tip, pt1, pt2, pt3, thresh)


                        bottom_point = dart_scorer_util.get_bottom_point(rest_pts[0], rest_pts[1])
                        cv2.line(img_roi, dart_tip, bottom_point, (0, 0, 255), 2)
                        cv2.line(thresh, dart_tip, bottom_point, (255, 0, 255), 2)

                        k = -0.215  # scaling factor for position adjustment of dart tip
                        vect = (dart_tip - bottom_point)
                        new_dart_tip = dart_tip + k * vect

                        cv2.circle(img_roi, new_dart_tip.astype(np.int32), 4, (0, 255, 0), -1)
                        new_radius, new_angle = dart_scorer_util.get_radius_and_angle(center_ellipse[0], center_ellipse[1], new_dart_tip[0], new_dart_tip[1])
                        new_val, new_mult = dart_scorer_util.evaluate_throw(new_radius, new_angle)

                        if len(scored_values) <= 20:
                            scored_values.append(new_val)
                            scored_mults.append(new_mult)
                        else:
                            update_dart_point = True
                            final_val = mode(scored_values)  # Take the most frequent result and use that as the final result
                            final_mult = mode(scored_mults)
                            values_of_round.append(final_val)
                            mults_of_round.append(final_mult)
                            default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)  # Reset the default image after every dart
                            if len(values_of_round) == 3:
                                self.reset_default_image_after_player()
                                self.enter_score_of_one_player(score1, score2)
                            scored_values = []
                            scored_mults = []

                    cv2.imshow("Threshold", thresh)

                    previous_img = img_roi
                    # TODO: Separate show image and processing image with cv2.copy
                    # cv2.ellipse(img_roi, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0, (255, 0, 0))
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_1 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_2 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_3 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_4 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_5 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_6 / 100)), (255, 0, 255), 1)
                    cv2.ellipse(img_roi, ellipse, (0, 255, 0), 2)

                    fps, img_roi = fpsReader.update(img_roi)
                    cv2.imshow("Dart Settings", utils.rez(img_roi, 1.5))
                else:
                    print("NO MARKERS FOUND!")
                    cv2.putText(img_undist, "NO MARKERS FOUND", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
                    cv2.imshow("Dart Settings", img_undist)
                    sleep(0.1)

                cv2.waitKey(1)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    cap.release()
                    exit()
            # UIFunctions.update_labels(window)

    def reset_default_image_after_player(self):
        """
        Resets the default image after a player has thrown 3 darts and triggers the corresponding gui functions
        :return:
        """
        global default_img
        UIFunctions.stop_detection_and_scoring(window)
        success, img = cap.read()  # Reset the default image after every dart
        if success:
            if USE_CAMERA_CALIBRATION_TO_UNDISTORT:
                img = utils.undistortFunction(img, meanMTX, meanDIST)
            else:
                img = img
        default_img = utils.reset_default_image(img, target_ROI_size, resize_for_squish)

    def enter_score_of_one_player(self, score1, score2):
        """
        Enters the score of one player into the dart scorer util
        :param score1:
        :param score2:
        :return:
        """
        global UNDO_LAST_FLAG, default_img, ACTIVE_PLAYER, values_of_round, mults_of_round
        UNDO_LAST_FLAG = False
        # if cv2.waitKey(0) & 0xFF == ord('\r'):
        # if window.ui.continue_button.isChecked():
        if not UNDO_LAST_FLAG:
            if not STOP_DETECTION:
                window.ui.press_enter_label.setText("")
            if ACTIVE_PLAYER == 1:
                dart_scorer_util.update_score(score1, values_of_round=values_of_round, mults_of_round=mults_of_round)
                ACTIVE_PLAYER = 2
            elif ACTIVE_PLAYER == 2:
                dart_scorer_util.update_score(score2, values_of_round=values_of_round, mults_of_round=mults_of_round)
                ACTIVE_PLAYER = 1
            values_of_round = []
            mults_of_round = []
        else:
            UNDO_LAST_FLAG = False

    def draw_detected_darts(self, dart_point, pt1, pt2, pt3, thresh):
        """
        Draw the detected darts on the threshold image as triangle and a dot indicating the dart tip
        :param dart_point: The point of the dart tip
        :param pt1: Triangle point 1
        :param pt2: Triangle point 2
        :param pt3: Triangle point 3
        :param thresh: the threshold image
        :return:
        """
        # Display the Dart point
        cv2.circle(thresh, dart_point, 4, (0, 0, 255), -1)
        # Display the triangles
        cv2.line(thresh, pt1.ravel(), pt2.ravel(), (255, 0, 255), 2)
        cv2.line(thresh, pt2.ravel(), pt3.ravel(), (255, 0, 255), 2)
        cv2.line(thresh, pt3.ravel(), pt1.ravel(), (255, 0, 255), 2)

    def prepare_differnce_image(self, TRIANGLE_DETECT_THRESH, difference):
        """
        Prepare the difference image for triangle detection, by applying a bilateral filter, gaussian blur and thresholding
        :param TRIANGLE_DETECT_THRESH:
        :param difference:
        :return:
        """
        blur = cv2.GaussianBlur(difference, (5, 5), 0)
        for i in range(10):
            blur = cv2.GaussianBlur(blur, (9, 9), 1)
        blur = cv2.bilateralFilter(blur, 9, 75, 75)
        ret, thresh = cv2.threshold(blur, TRIANGLE_DETECT_THRESH, 255, 0)
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        return gray, thresh


def detect_dart_circle_and_set_limits(img_roi):
    # cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automaticMode, threshold_new = opencv_gui_sliders.updateTrackBar()
    cannyLow = 80
    cannyHigh = 160
    noGauss = 2
    minArea = 800
    erosions = 1
    dilations = 1
    epsilon = 5 / 1000
    showFilters = 0
    global contours, radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, cnt, ellipse, x, y, a, b, angle, center_ellipse, x_offset_current, \
        y_offset_current, TRIANGLE_DETECT_THRESH, DARTBOARD_AREA, current_settings
    imgContours, contours, imgCanny = ContourUtils.get_contours(img=img_roi, cThr=(cannyLow, cannyHigh),
                                                                gaussFilters=noGauss, minArea=minArea,
                                                                epsilon=epsilon, draw=False,
                                                                erosions=erosions, dilations=dilations,
                                                                showFilters=showFilters)
    radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset = opencv_gui_sliders.update_dart_trackbars()
    new_settings = [radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset]
    image_area = img_roi.shape[0] * img_roi.shape[1]
    contours = [cnt for cnt in contours if image_area * 0.5 < cnt[1] < image_area * 0.9]  # Filter out contours that are too small or too big
    # get biggest contour
    cnt = get_biggest_contour(contours)
    if cnt is None:
        return
    # Create the outermost Circle
    # if a radius changed
    if ellipse is None or new_settings != current_settings:  # Save the outermost ellipse for later to avoid useless re calculation !
        print("Recalculating the outermost ellipse")
        radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset = new_settings
        ellipse = cv2.fitEllipse(cnt[4])  # Also a benefit for stability of the outer ellipse --> not jumping from frame to frame
        # get area of ellipse
        DARTBOARD_AREA = cv2.contourArea(cnt[4])
        x, y = ellipse[0]
        a, b = ellipse[1]
        angle = ellipse[2]
        center_ellipse = (int(x + x_offset / 10), int(y + y_offset / 10))
        a = a / 2
        b = b / 2
        # set the limits also only once
        dart_scorer_util.bullsLimit = a * (radius_1 / 100)
        dart_scorer_util.singleBullsLimit = a * (radius_2 / 100)
        dart_scorer_util.innerTripleLimit = a * (radius_3 / 100)
        dart_scorer_util.outerTripleLimit = a * (radius_4 / 100)
        dart_scorer_util.innerDoubleLimit = a * (radius_5 / 100)
        dart_scorer_util.outerBoardLimit = a * (radius_6 / 100)
        current_settings = [radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset]
        print("Limits set")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    label_update_timer = QtCore.QTimer()
    label_update_timer.timeout.connect(window.update_labels)
    label_update_timer.start(10)  # every 10 milliseconds

    try:
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists('calibration_data.json'):
            os.remove('calibration_data.json')
        if os.path.exists('calibration_data.pkl'):
            os.remove('calibration_data.pkl')
        sys.exit(1)
