"""
Admin Database Module
Handles candidate data persistence, evidence storage, and automated cleanup.
"""

import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path


class AdminDatabase:
    def __init__(self):
        base = os.path.dirname(os.path.abspath(__file__))
        self.candidates_db_path = os.path.join(base, "admin_data", "candidates.json")
        self.evidence_folder = os.path.join(base, "evidence")
        self.logs_folder = os.path.join(base, "logs")
        self.uploads_folder = os.path.join(base, "uploads")
        self.admin_logs_path = os.path.join(base, "admin_data", "admin_logs.json")
        self.cleanup_hours = 24  # Delete logs older than 24 hours

        # Create directories
        os.makedirs(os.path.join(base, "admin_data"), exist_ok=True)
        os.makedirs(self.evidence_folder, exist_ok=True)
        
        # Initialize databases
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure candidate database exists."""
        if not os.path.exists(self.candidates_db_path):
            with open(self.candidates_db_path, 'w') as f:
                json.dump({}, f, indent=2)
        
        if not os.path.exists(self.admin_logs_path):
            with open(self.admin_logs_path, 'w') as f:
                json.dump([], f, indent=2)
    
    def get_all_candidates(self):
        """Get all candidates from database."""
        try:
            with open(self.candidates_db_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def get_candidate(self, candidate_id):
        """Get specific candidate data."""
        candidates = self.get_all_candidates()
        return candidates.get(candidate_id)
    
    def save_candidate(self, candidate_id, candidate_data):
        """Save or update candidate data."""
        candidates = self.get_all_candidates()
        candidate_data['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        candidates[candidate_id] = candidate_data
        
        with open(self.candidates_db_path, 'w') as f:
            json.dump(candidates, f, indent=2)
        
        self._log_admin_action('save_candidate', candidate_id, f"Updated candidate: {candidate_id}")
    
    def update_candidate(self, candidate_id, updates):
        """Update specific candidate fields."""
        candidate = self.get_candidate(candidate_id)
        if candidate:
            candidate.update(updates)
            candidate['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save_candidate(candidate_id, candidate)
            return True
        return False
    
    def delete_candidate(self, candidate_id):
        """Delete candidate and associated data, preserving reference photo as evidence."""
        candidates = self.get_all_candidates()
        if candidate_id in candidates:
            del candidates[candidate_id]

            with open(self.candidates_db_path, 'w') as f:
                json.dump(candidates, f, indent=2)

            # Delete interview logs
            log_path = os.path.join(self.logs_folder, f"{candidate_id}.json")
            if os.path.exists(log_path):
                os.remove(log_path)

            # Delete evidence folder
            evidence_path = os.path.join(self.evidence_folder, candidate_id)
            if os.path.exists(evidence_path):
                shutil.rmtree(evidence_path)

            # Delete reference photo if it still exists
            ref_path = os.path.join(self.uploads_folder, f"reference_{candidate_id}.jpg")
            if os.path.exists(ref_path):
                try:
                    os.remove(ref_path)
                except Exception as e:
                    print(f"[CLEANUP] Error deleting reference photo: {e}")

            # Remove last frame if present
            last_frame_path = os.path.join(self.uploads_folder, f"last_frame_{candidate_id}.jpg")
            if os.path.exists(last_frame_path):
                try:
                    os.remove(last_frame_path)
                except Exception as e:
                    print(f"[CLEANUP] Error deleting last frame: {e}")

            self._log_admin_action('delete_candidate', candidate_id, f"Deleted candidate: {candidate_id}")
            return True
        return False
    
    def store_evidence(self, candidate_id, evidence_image_path, evidence_name=None):
        """Store evidence photo for a candidate."""
        if not os.path.exists(evidence_image_path):
            print(f"[EVIDENCE] Source file not found: {evidence_image_path}")
            return False

        # Create evidence folder for candidate
        evidence_path = os.path.join(self.evidence_folder, candidate_id)
        os.makedirs(evidence_path, exist_ok=True)

        # Save evidence with timestamp to avoid overwrites
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(evidence_name)[0] if evidence_name else "evidence"
        filename = f"{base_name}_{timestamp}.jpg"
        dest_path = os.path.join(evidence_path, filename)

        try:
            shutil.copy2(evidence_image_path, dest_path)
        except Exception as e:
            print(f"[EVIDENCE] Failed to copy to {dest_path}: {e}")
            return False

        # Update candidate record
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            candidate = {}

        if 'evidence_photos' not in candidate:
            candidate['evidence_photos'] = []

        candidate['evidence_photos'].append({
            'filename': filename,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'path': dest_path
        })

        self.save_candidate(candidate_id, candidate)
        print(f"[EVIDENCE] Stored: {dest_path}")
        return True

    def delete_reference_photo(self, candidate_id):
        """Delete the reference photo from the uploads folder."""
        ref_path = os.path.join(self.uploads_folder, f"reference_{candidate_id}.jpg")
        if os.path.exists(ref_path):
            try:
                os.remove(ref_path)
                return True
            except Exception as e:
                print(f"[CLEANUP] Error deleting reference photo: {e}")
        return False
    
    def get_evidence_photos(self, candidate_id):
        """Get all evidence photos for a candidate."""
        candidate = self.get_candidate(candidate_id)
        if candidate:
            return candidate.get('evidence_photos', [])
        return []
    
    def delete_old_logs(self):
        """Delete logs older than 24 hours."""
        deleted_count = 0
        if not os.path.exists(self.logs_folder):
            return deleted_count
        
        now = time.time()
        cleanup_seconds = self.cleanup_hours * 3600
        
        for filename in os.listdir(self.logs_folder):
            log_path = os.path.join(self.logs_folder, filename)
            if os.path.isfile(log_path):
                file_age = now - os.path.getmtime(log_path)
                if file_age > cleanup_seconds:
                    os.remove(log_path)
                    deleted_count += 1
                    self._log_admin_action('auto_cleanup', filename.replace('.json', ''), 
                                         f"Deleted log file: {filename}")
        
        return deleted_count
    
    def get_logs_older_than_24h(self):
        """Get list of logs that are older than 24 hours."""
        old_logs = []
        if not os.path.exists(self.logs_folder):
            return old_logs
        
        now = time.time()
        cleanup_seconds = 24 * 3600
        
        for filename in os.listdir(self.logs_folder):
            log_path = os.path.join(self.logs_folder, filename)
            if os.path.isfile(log_path) and filename.endswith('.json'):
                file_age = now - os.path.getmtime(log_path)
                if file_age > cleanup_seconds:
                    old_logs.append({
                        'filename': filename,
                        'candidate_id': filename.replace('.json', ''),
                        'age_hours': round(file_age / 3600, 1),
                        'path': log_path
                    })
        
        return old_logs
    
    def can_modify_candidate(self, candidate_id):
        """Check if candidate can be modified (must be registered > 24h)."""
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            return False
        
        created_at_str = candidate.get('created_at', candidate.get('loaded_at'))
        if not created_at_str:
            return False
        
        try:
            created_time = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            time_diff = datetime.now() - created_time
            return time_diff >= timedelta(hours=24)
        except:
            return False
    
    def _log_admin_action(self, action, target, details):
        """Log admin actions for audit trail."""
        try:
            with open(self.admin_logs_path, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
        
        logs.append({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'action': action,
            'target': target,
            'details': details
        })
        
        with open(self.admin_logs_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_admin_logs(self, limit=100):
        """Get admin action logs."""
        try:
            with open(self.admin_logs_path, 'r') as f:
                logs = json.load(f)
            return logs[-limit:]
        except:
            return []
    
    def get_interview_logs(self, candidate_id):
        """Get interview verification logs for a candidate."""
        log_path = os.path.join(self.logs_folder, f"{candidate_id}.json")
        
        if not os.path.exists(log_path):
            return []
        
        try:
            with open(log_path, 'r') as f:
                logs = json.load(f)
            return logs
        except:
            return []
    
    def get_interview_summary(self, candidate_id):
        """Get summary statistics for candidate interview."""
        logs = self.get_interview_logs(candidate_id)
        
        if not logs:
            return {
                'candidate_id': candidate_id,
                'total_checks': 0,
                'verified_checks': 0,
                'failed_checks': 0,
                'warnings': 0,
                'integrity_score': 0,
                'status': 'no_data'
            }
        
        total = len(logs)
        verified = sum(1 for log in logs if log.get('verified') is True)
        failed = sum(1 for log in logs if log.get('verified') is False)
        warnings = sum(1 for log in logs if 'warning' in log.get('status', '').lower())
        terminated = any(log.get('terminate') for log in logs)
        
        return {
            'candidate_id': candidate_id,
            'total_checks': total,
            'verified_checks': verified,
            'failed_checks': failed,
            'warning_count': warnings,
            'integrity_score': round((verified / total * 100) if total > 0 else 0, 1),
            'status': 'Terminated' if terminated else 'Active',
            'first_check': logs[0]['timestamp'] if logs else None,
            'last_check': logs[-1]['timestamp'] if logs else None
        }


# Global instance
admin_db = AdminDatabase()
