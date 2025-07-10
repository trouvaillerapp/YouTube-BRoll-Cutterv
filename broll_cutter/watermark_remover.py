"""
Advanced watermark removal module using computer vision and inpainting techniques
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

@dataclass
class WatermarkRegion:
    """Represents a watermark region in the video"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    region_type: str = "logo"  # logo, text, bug, overlay
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y, 
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence,
            'region_type': self.region_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WatermarkRegion':
        return cls(
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            confidence=data.get('confidence', 0.0),
            region_type=data.get('region_type', 'logo')
        )

class WatermarkRemover:
    """Advanced watermark detection and removal using multiple CV techniques"""
    
    def __init__(self,
                 detection_threshold: float = 0.8,
                 min_region_size: int = 20,
                 max_region_size: int = 300,
                 inpaint_method: str = "telea",
                 enable_auto_detection: bool = True,
                 temporal_consistency: bool = True):
        """
        Initialize the watermark remover
        
        Args:
            detection_threshold: Confidence threshold for watermark detection
            min_region_size: Minimum size for watermark regions (pixels)
            max_region_size: Maximum size for watermark regions (pixels)
            inpaint_method: Inpainting algorithm ("telea" or "ns")
            enable_auto_detection: Enable automatic watermark detection
            temporal_consistency: Ensure consistency across frames
        """
        self.detection_threshold = detection_threshold
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size
        self.enable_auto_detection = enable_auto_detection
        self.temporal_consistency = temporal_consistency
        
        # Inpainting method selection
        self.inpaint_methods = {
            "telea": cv2.INPAINT_TELEA,
            "ns": cv2.INPAINT_NS
        }
        self.inpaint_method = self.inpaint_methods.get(inpaint_method, cv2.INPAINT_TELEA)
        
        # Common watermark positions (normalized coordinates)
        self.common_positions = [
            {"x": 0.02, "y": 0.02, "w": 0.2, "h": 0.1},     # Top-left
            {"x": 0.78, "y": 0.02, "w": 0.2, "h": 0.1},     # Top-right
            {"x": 0.02, "y": 0.88, "w": 0.2, "h": 0.1},     # Bottom-left
            {"x": 0.78, "y": 0.88, "w": 0.2, "h": 0.1},     # Bottom-right
            {"x": 0.4, "y": 0.02, "w": 0.2, "h": 0.08},     # Top-center
            {"x": 0.4, "y": 0.9, "w": 0.2, "h": 0.08},      # Bottom-center
        ]
    
    def detect_watermarks(self, video_path: str, sample_frames: int = 10) -> List[WatermarkRegion]:
        """
        Automatically detect watermarks in video
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to analyze for detection
            
        Returns:
            List of detected watermark regions
        """
        try:
            logger.info(f"Starting watermark detection for: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Sample frames evenly throughout the video
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            
            detected_regions = []
            sample_frames_data = []
            
            for frame_idx in tqdm(frame_indices, desc="Analyzing frames for watermarks"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    sample_frames_data.append(frame)
            
            cap.release()
            
            if len(sample_frames_data) < 2:
                logger.warning("Insufficient frames for watermark detection")
                return []
            
            # Detect static regions (potential watermarks)
            static_regions = self._detect_static_regions(sample_frames_data)
            
            # Detect logo-like regions
            logo_regions = self._detect_logo_regions(sample_frames_data[0], frame_width, frame_height)
            
            # Combine and filter detections
            all_regions = static_regions + logo_regions
            filtered_regions = self._filter_and_merge_regions(all_regions, frame_width, frame_height)
            
            logger.info(f"Detected {len(filtered_regions)} watermark regions")
            return filtered_regions
            
        except Exception as e:
            logger.error(f"Watermark detection failed: {str(e)}")
            return []
    
    def _detect_static_regions(self, frames: List[np.ndarray]) -> List[WatermarkRegion]:
        """Detect regions that remain static across multiple frames"""
        
        if len(frames) < 2:
            return []
        
        # Calculate frame differences
        diff_accumulator = np.zeros(frames[0].shape[:2], dtype=np.float32)
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            diff_accumulator += diff.astype(np.float32)
        
        # Normalize
        diff_accumulator /= (len(frames) - 1)
        
        # Threshold to find static regions
        static_mask = (diff_accumulator < 10).astype(np.uint8) * 255
        
        # Find contours of static regions
        contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size
            if self.min_region_size * self.min_region_size < area < self.max_region_size * self.max_region_size:
                # Check if region is in common watermark positions
                confidence = self._calculate_position_confidence(x, y, w, h, frames[0].shape[1], frames[0].shape[0])
                
                if confidence > 0.3:  # Minimum confidence for static regions
                    region = WatermarkRegion(
                        x=x, y=y, width=w, height=h,
                        confidence=confidence,
                        region_type="static"
                    )
                    regions.append(region)
        
        return regions
    
    def _detect_logo_regions(self, frame: np.ndarray, frame_width: int, frame_height: int) -> List[WatermarkRegion]:
        """Detect logo-like regions using edge detection and contour analysis"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size and aspect ratio
            if (self.min_region_size < w < self.max_region_size and 
                self.min_region_size < h < self.max_region_size and
                0.3 < w/h < 3.0):  # Reasonable aspect ratio
                
                # Calculate confidence based on position and characteristics
                position_conf = self._calculate_position_confidence(x, y, w, h, frame_width, frame_height)
                edge_conf = self._calculate_edge_confidence(gray[y:y+h, x:x+w])
                
                confidence = (position_conf + edge_conf) / 2
                
                if confidence > 0.4:
                    region = WatermarkRegion(
                        x=x, y=y, width=w, height=h,
                        confidence=confidence,
                        region_type="logo"
                    )
                    regions.append(region)
        
        return regions
    
    def _calculate_position_confidence(self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> float:
        """Calculate confidence based on position relative to common watermark locations"""
        
        # Normalize coordinates
        norm_x = x / frame_w
        norm_y = y / frame_h
        norm_w = w / frame_w
        norm_h = h / frame_h
        
        max_confidence = 0.0
        
        for pos in self.common_positions:
            # Calculate overlap with common position
            overlap_x = max(0, min(norm_x + norm_w, pos["x"] + pos["w"]) - max(norm_x, pos["x"]))
            overlap_y = max(0, min(norm_y + norm_h, pos["y"] + pos["h"]) - max(norm_y, pos["y"]))
            overlap_area = overlap_x * overlap_y
            
            # Calculate confidence based on overlap
            region_area = norm_w * norm_h
            if region_area > 0:
                confidence = overlap_area / region_area
                max_confidence = max(max_confidence, confidence)
        
        return max_confidence
    
    def _calculate_edge_confidence(self, region: np.ndarray) -> float:
        """Calculate confidence based on edge characteristics of the region"""
        
        if region.size == 0:
            return 0.0
        
        # Calculate edge density
        edges = cv2.Canny(region, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = region.size
        edge_density = edge_pixels / total_pixels
        
        # Logo-like regions typically have moderate edge density
        if 0.05 < edge_density < 0.3:
            return min(1.0, edge_density * 3.0)
        else:
            return 0.1
    
    def _filter_and_merge_regions(self, regions: List[WatermarkRegion], frame_w: int, frame_h: int) -> List[WatermarkRegion]:
        """Filter overlapping regions and merge nearby ones"""
        
        if not regions:
            return []
        
        # Sort by confidence
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        filtered_regions = []
        
        for region in regions:
            # Check if this region overlaps significantly with any existing region
            should_add = True
            
            for existing in filtered_regions:
                overlap = self._calculate_region_overlap(region, existing)
                
                if overlap > 0.5:  # Significant overlap
                    should_add = False
                    break
            
            if should_add and region.confidence > self.detection_threshold:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def _calculate_region_overlap(self, region1: WatermarkRegion, region2: WatermarkRegion) -> float:
        """Calculate overlap ratio between two regions"""
        
        x1_max = max(region1.x, region2.x)
        y1_max = max(region1.y, region2.y)
        x2_min = min(region1.x + region1.width, region2.x + region2.width)
        y2_min = min(region1.y + region1.height, region2.y + region2.height)
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        overlap_area = (x2_min - x1_max) * (y2_min - y1_max)
        region1_area = region1.width * region1.height
        region2_area = region2.width * region2.height
        
        # Return overlap ratio relative to smaller region
        smaller_area = min(region1_area, region2_area)
        return overlap_area / smaller_area if smaller_area > 0 else 0.0
    
    def remove_watermarks(self, video_path: str, output_path: str, 
                         watermark_regions: List[WatermarkRegion] = None,
                         progress_callback: Optional[callable] = None) -> str:
        """
        Remove watermarks from video
        
        Args:
            video_path: Input video path
            output_path: Output video path
            watermark_regions: List of watermark regions to remove
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to processed video
        """
        try:
            logger.info(f"Starting watermark removal for: {video_path}")
            
            # Auto-detect watermarks if not provided
            if watermark_regions is None and self.enable_auto_detection:
                watermark_regions = self.detect_watermarks(video_path)
            
            if not watermark_regions:
                logger.warning("No watermark regions specified or detected")
                # Just copy the video if no watermarks to remove
                import shutil
                shutil.copy2(video_path, output_path)
                return output_path
            
            # Process video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            # Create combined mask for all watermark regions
            mask = self._create_watermark_mask(watermark_regions, frame_width, frame_height)
            
            # Process frames
            frame_count = 0
            with tqdm(total=total_frames, desc="Removing watermarks") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Remove watermarks from frame
                    processed_frame = self._remove_watermarks_from_frame(frame, mask, watermark_regions)
                    
                    # Write processed frame
                    out.write(processed_frame)
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # Update progress
                    if progress_callback and frame_count % 30 == 0:  # Update every 30 frames
                        progress = (frame_count / total_frames) * 100
                        progress_callback({'step': 'watermark_removal', 'progress': progress})
            
            # Cleanup
            cap.release()
            out.release()
            
            logger.info(f"Watermark removal completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Watermark removal failed: {str(e)}")
            raise
    
    def _create_watermark_mask(self, regions: List[WatermarkRegion], width: int, height: int) -> np.ndarray:
        """Create a binary mask for all watermark regions"""
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for region in regions:
            # Ensure coordinates are within frame bounds
            x1 = max(0, region.x)
            y1 = max(0, region.y)
            x2 = min(width, region.x + region.width)
            y2 = min(height, region.y + region.height)
            
            # Add region to mask
            mask[y1:y2, x1:x2] = 255
        
        # Apply morphological operations to smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _remove_watermarks_from_frame(self, frame: np.ndarray, mask: np.ndarray, 
                                    regions: List[WatermarkRegion]) -> np.ndarray:
        """Remove watermarks from a single frame using inpainting"""
        
        if np.sum(mask) == 0:  # No watermarks to remove
            return frame
        
        # Apply inpainting
        inpainted = cv2.inpaint(frame, mask, 3, self.inpaint_method)
        
        # Post-process to improve quality
        if self.temporal_consistency:
            # Apply slight blur to inpainted regions to reduce artifacts
            blur_mask = mask / 255.0
            blur_mask = np.stack([blur_mask] * 3, axis=2)
            
            blurred = cv2.GaussianBlur(inpainted, (3, 3), 0)
            inpainted = (blurred * blur_mask + inpainted * (1 - blur_mask)).astype(np.uint8)
        
        return inpainted
    
    def save_watermark_regions(self, regions: List[WatermarkRegion], output_path: str):
        """Save detected watermark regions to JSON file"""
        
        data = {
            'total_regions': len(regions),
            'detection_params': {
                'threshold': self.detection_threshold,
                'min_size': self.min_region_size,
                'max_size': self.max_region_size
            },
            'regions': [region.to_dict() for region in regions]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Watermark regions saved to: {output_path}")
    
    def load_watermark_regions(self, input_path: str) -> List[WatermarkRegion]:
        """Load watermark regions from JSON file"""
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        regions = [WatermarkRegion.from_dict(region_dict) for region_dict in data['regions']]
        
        logger.info(f"Loaded {len(regions)} watermark regions from: {input_path}")
        return regions
    
    def preview_watermark_detection(self, video_path: str, output_image_path: str):
        """Create a preview image showing detected watermark regions"""
        
        # Detect watermarks
        regions = self.detect_watermarks(video_path, sample_frames=5)
        
        # Get a frame to draw on
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Could not read frame from video")
        
        # Draw detected regions
        preview = frame.copy()
        for i, region in enumerate(regions):
            # Draw rectangle
            cv2.rectangle(preview, 
                         (region.x, region.y), 
                         (region.x + region.width, region.y + region.height),
                         (0, 0, 255), 2)
            
            # Add label
            label = f"{region.region_type} ({region.confidence:.2f})"
            cv2.putText(preview, label, 
                       (region.x, region.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save preview
        cv2.imwrite(output_image_path, preview)
        logger.info(f"Watermark detection preview saved to: {output_image_path}")
        
        return regions