import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

class DetectionMetrics:
    """
    Comprehensive metrics for object detection evaluation
    """
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_map(predictions: List[Dict], ground_truth: List[Dict], 
                     iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate Mean Average Precision (mAP) for object detection
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            iou_threshold: IoU threshold for considering a detection as correct
            
        Returns:
            Dictionary with mAP scores per class and overall mAP
        """
        # This is a simplified implementation
        # In practice, you'd use libraries like pycocotools for more accurate mAP calculation
        
        class_aps = {}
        unique_classes = set()
        
        # Collect all unique classes
        for pred in predictions:
            unique_classes.add(pred['class_id'])
        for gt in ground_truth:
            unique_classes.add(gt['class_id'])
        
        for class_id in unique_classes:
            # Filter predictions and ground truth for this class
            class_preds = [p for p in predictions if p['class_id'] == class_id]
            class_gt = [g for g in ground_truth if g['class_id'] == class_id]
            
            if not class_gt:
                class_aps[class_id] = 0.0
                continue
            
            # Sort predictions by confidence
            class_preds.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate precision and recall
            tp = 0
            fp = 0
            matched_gt = set()
            
            precisions = []
            recalls = []
            
            for pred in class_preds:
                best_iou = 0.0
                best_gt_idx = -1
                
                for i, gt in enumerate(class_gt):
                    if i in matched_gt:
                        continue
                    iou = DetectionMetrics.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / len(class_gt)
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Calculate Average Precision using the precision-recall curve
            if precisions and recalls:
                ap = average_precision_score([1] * len(recalls), precisions)
            else:
                ap = 0.0
            
            class_aps[class_id] = ap
        
        # Calculate mean AP
        mean_ap = np.mean(list(class_aps.values())) if class_aps else 0.0
        
        return {
            'mAP': mean_ap,
            'class_aps': class_aps
        }
    
    @staticmethod
    def generate_confusion_matrix(predictions: List[Dict], ground_truth: List[Dict],
                                 class_names: List[str]) -> np.ndarray:
        """Generate confusion matrix for multi-class detection"""
        n_classes = len(class_names)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        # This would need more sophisticated matching logic
        # For now, this is a placeholder implementation
        
        return confusion_matrix