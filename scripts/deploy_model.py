#!/usr/bin/env python3
"""
Safe Model Deployment Script
============================

Safely deploys models from models_dev/ to models/ (production).

Features:
- Pre-deployment validation
- Automatic backup of current production model
- Git integration (commit + tag)
- Rollback capability
- Deployment log

Usage:
    # Deploy a single model
    python3 scripts/deploy_model.py weinstein_core_v2

    # Deploy with custom description
    python3 scripts/deploy_model.py weinstein_core_v2 --description "Improved confidence filters"

    # Deploy multiple models
    python3 scripts/deploy_model.py weinstein_core_v2 rsi_mean_reversion_v3

    # Dry run (see what would happen without actually deploying)
    python3 scripts/deploy_model.py weinstein_core_v2 --dry-run

    # Skip git commit
    python3 scripts/deploy_model.py weinstein_core_v2 --no-git
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DEV_DIR = "models_dev"
MODELS_PROD_DIR = "models"
BACKUP_DIR = "models_backup"
DEPLOYMENT_LOG = "deployment_log.txt"

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_dev_model(model_filename):
    """Validate that development model file exists and is valid Python"""
    filepath = os.path.join(MODELS_DEV_DIR, f"{model_filename}.py")
    
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found!")
        return False
    
    # Try to compile the file to check for syntax errors
    try:
        with open(filepath, 'r') as f:
            code = f.read()
            compile(code, filepath, 'exec')
        print(f"✅ {model_filename}.py: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}:")
        print(f"   Line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error validating {filepath}: {e}")
        return False


def determine_target_filename(dev_filename):
    """
    Determine production filename from development filename
    
    Examples:
        weinstein_core_v2 -> weinstein_core
        rsi_mean_reversion_v3 -> rsi_mean_reversion
        new_momentum_model -> new_momentum_model (no change)
    """
    # Remove version suffix (v2, v3, etc.)
    if '_v' in dev_filename:
        base_name = dev_filename.rsplit('_v', 1)[0]
        version = dev_filename.rsplit('_v', 1)[1]
        
        # Verify it looks like a version number
        if version.isdigit():
            return base_name
    
    # If no version suffix, use as-is
    return dev_filename


def check_prod_model_exists(prod_filename):
    """Check if production model already exists"""
    filepath = os.path.join(MODELS_PROD_DIR, f"{prod_filename}.py")
    return os.path.exists(filepath)


# ============================================================================
# BACKUP FUNCTIONS
# ============================================================================

def create_backup(prod_filename):
    """Create backup of current production model"""
    source = os.path.join(MODELS_PROD_DIR, f"{prod_filename}.py")
    
    if not os.path.exists(source):
        print(f"ℹ️  No existing production model to backup")
        return None
    
    # Create backup directory if needed
    Path(BACKUP_DIR).mkdir(exist_ok=True)
    
    # Backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{prod_filename}_{timestamp}.py"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    
    # Copy file
    shutil.copy2(source, backup_path)
    
    print(f"✅ Backup created: {backup_path}")
    return backup_path


# ============================================================================
# DEPLOYMENT FUNCTIONS
# ============================================================================

def deploy_model(dev_filename, prod_filename, dry_run=False):
    """Deploy model from dev to production"""
    source = os.path.join(MODELS_DEV_DIR, f"{dev_filename}.py")
    target = os.path.join(MODELS_PROD_DIR, f"{prod_filename}.py")
    
    if dry_run:
        print(f"[DRY RUN] Would copy: {source} -> {target}")
        return True
    
    try:
        shutil.copy2(source, target)
        print(f"✅ Deployed: {source} -> {target}")
        return True
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False


# ============================================================================
# GIT INTEGRATION
# ============================================================================

def check_git_status():
    """Check if git is available and repo is clean"""
    try:
        # Check if git is available
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "Git not found"
        
        # Check if in git repo
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "Not in a git repository"
        
        # Check status
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        uncommitted = [line for line in result.stdout.split('\n') if line.strip()]
        
        return True, uncommitted
        
    except Exception as e:
        return False, str(e)


def git_commit_deployment(deployed_models, description="Model deployment"):
    """Commit deployment to git"""
    try:
        # Add models directory
        subprocess.run(['git', 'add', MODELS_PROD_DIR], check=True)
        
        # Create commit message
        commit_msg = f"Deploy models to production: {', '.join(deployed_models)}\n\n{description}"
        
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        
        print(f"✅ Git commit created")
        
        # Optionally create tag
        tag_name = f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        subprocess.run(['git', 'tag', '-a', tag_name, '-m', description], check=True)
        
        print(f"✅ Git tag created: {tag_name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")
        return False


# ============================================================================
# LOGGING
# ============================================================================

def log_deployment(deployed_models, backups, description):
    """Log deployment to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"""
{'='*80}
DEPLOYMENT: {timestamp}
{'='*80}
Models deployed: {', '.join(deployed_models)}
Description: {description}
Backups: {', '.join(backups) if backups else 'None (new models)'}
{'='*80}

"""
    
    with open(DEPLOYMENT_LOG, 'a') as f:
        f.write(log_entry)
    
    print(f"✅ Deployment logged to {DEPLOYMENT_LOG}")


# ============================================================================
# INTERACTIVE CONFIRMATION
# ============================================================================

def confirm_deployment(deployments):
    """Ask user to confirm deployment"""
    print("\n" + "="*80)
    print("DEPLOYMENT PLAN")
    print("="*80)
    
    for dev_file, prod_file, is_new in deployments:
        action = "CREATE NEW" if is_new else "REPLACE"
        print(f"{action:12s} {prod_file}.py")
        print(f"             Source: models_dev/{dev_file}.py")
        print()
    
    print("="*80)
    
    response = input("\nProceed with deployment? [yes/no]: ").strip().lower()
    return response in ['yes', 'y']


# ============================================================================
# MAIN DEPLOYMENT WORKFLOW
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deploy models to production')
    parser.add_argument('models', nargs='+', help='Model filenames (without .py) to deploy')
    parser.add_argument('--description', type=str, default='', help='Deployment description')
    parser.add_argument('--dry-run', action='store_true', help='Show what would happen without deploying')
    parser.add_argument('--no-git', action='store_true', help='Skip git commit')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MODEL DEPLOYMENT TO PRODUCTION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to deploy: {', '.join(args.models)}")
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No actual changes will be made")
    print("="*80)
    
    # Validate all models first
    print("\n📋 VALIDATION PHASE")
    print("-" * 80)
    
    all_valid = True
    deployments = []
    
    for model_name in args.models:
        # Validate dev model
        if not validate_dev_model(model_name):
            all_valid = False
            continue
        
        # Determine target filename
        target_name = determine_target_filename(model_name)
        is_new = not check_prod_model_exists(target_name)
        
        if is_new:
            print(f"ℹ️  {target_name}.py: New model (will be created)")
        else:
            print(f"ℹ️  {target_name}.py: Existing model (will be replaced)")
        
        deployments.append((model_name, target_name, is_new))
    
    if not all_valid:
        print("\n❌ Validation failed for one or more models")
        return 1
    
    print("-" * 80)
    print(f"✅ All {len(deployments)} model(s) validated")
    
    # Confirm deployment
    if not args.force and not args.dry_run:
        if not confirm_deployment(deployments):
            print("\n❌ Deployment cancelled by user")
            return 0
    
    # Create backups and deploy
    print("\n🚀 DEPLOYMENT PHASE")
    print("-" * 80)
    
    backups = []
    deployed = []
    
    for dev_file, prod_file, is_new in deployments:
        # Create backup if replacing existing
        if not is_new:
            backup_path = create_backup(prod_file)
            if backup_path:
                backups.append(backup_path)
        
        # Deploy
        if deploy_model(dev_file, prod_file, args.dry_run):
            deployed.append(prod_file)
        else:
            print(f"❌ Deployment failed for {dev_file}")
            return 1
    
    print("-" * 80)
    
    if args.dry_run:
        print("\n✅ DRY RUN COMPLETE - No actual changes made")
        return 0
    
    print(f"\n✅ Successfully deployed {len(deployed)} model(s)")
    
    # Log deployment
    print("\n📝 LOGGING PHASE")
    print("-" * 80)
    
    log_deployment(deployed, backups, args.description or "Model deployment")
    
    # Git commit
    if not args.no_git:
        print("\n📦 GIT INTEGRATION")
        print("-" * 80)
        
        git_ok, git_status = check_git_status()
        
        if git_ok:
            if git_commit_deployment(deployed, args.description or "Model deployment"):
                print("\n✅ Git commit and tag created")
                print("   Don't forget to push: git push origin main --tags")
            else:
                print("\n⚠️  Git commit failed (not critical)")
        else:
            print(f"⚠️  Git not available: {git_status}")
    
    # Final summary
    print("\n" + "="*80)
    print("DEPLOYMENT COMPLETE")
    print("="*80)
    print(f"Models deployed: {', '.join(deployed)}")
    if backups:
        print(f"Backups created: {len(backups)}")
        print(f"Backup location: {BACKUP_DIR}/")
    print("\nNext steps:")
    print("1. ✅ Models are now in production")
    print("2. 🔄 Restart live trading system (if running)")
    print("3. 📊 Monitor first signals tomorrow morning")
    print("4. 📝 Update CHANGELOG.md with deployment details")
    if not args.no_git:
        print("5. 📤 Push to git: git push origin main --tags")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
