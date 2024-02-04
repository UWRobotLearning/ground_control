from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import from_repo_root

import os
import subprocess
from git import Repo, Actor
from git.exc import InvalidGitRepositoryError
import logging

# Creates/saves commit hashes in this repo (autocommitting changes if needed)
# and in the log repo (for codesaving). For commits in this repo:
# - If autocommit is enabled and new changes are made, autocommits and saves the new commit's hash.
# - Otherwise, saves the latest commit hash. If force_manual_commit is enabled, forces user to
#   commit all work manually before running.
# For commits in the log repo, if codesave_to_logs is enabled, creates a new commit and saves it
# to the logs of the current experiment. Otherwise saves an empty hash. Both commits are saved in
# the "code_commits.txt" file in the experiment log folder.
def handle_codesave(cfg):
    latest_commit_hash, codesave_autocommit_hash = "", ""
    log = logging.getLogger(__name__)
    # Handle latest hash for this (main, development) repo
    if cfg.autocommit:
        latest_commit_hash = autocommit(commit_message=cfg.autocommit_message, push=cfg.autocommit_push)
    else:
        latest_commit_hash = check_commit(cfg.force_manual_commit)
        log.info(f"Didn't autocommit to development repo, latest commit: {latest_commit_hash}")
    # Handle commit hash for codesave repo if enabled (otherwise stays empty)
    if cfg.codesave_to_logs:
        codesave_autocommit_hash = codesave_in_logs(logs_root=cfg.log_dir,
                                                    codesave_dir=cfg.codesave_dir_in_logs,
                                                    commit_message=cfg.autocommit_message,
                                                    push=cfg.codesave_push)
    # Save both hashes to the log repo (which the script should be in due to Hydra,
    # need to change this path if that changes in the future)
    with open('code_commits.txt', 'a') as f:
        f.write(f'latest_commit_hash: {latest_commit_hash}\n')
        f.write(f'codesave_autocommit_hash: {codesave_autocommit_hash}\n')
    log.info("Successfully saved relevant commits in the experiment log folder!")
    return latest_commit_hash, codesave_autocommit_hash

# Checks if all changes to this repo have been committed, including untracked files. 
# If force_commit is enabled and either a git repo is not initialized or there are
# uncommitted changes, raises an exception, if force_commit is disabled then returns None.
# Otherwise, returns the latest commit hash.
def check_commit(force_commit=False):
    try:
        repo = Repo(LEGGED_GYM_ROOT_DIR)
    except InvalidGitRepositoryError:
        if force_commit:
            raise Exception("force_commit: ground_control is not a git repo! Please run \'git init\' " + 
                            "and commit all changes before running this script.")
        return ""
    
    if repo.is_dirty(untracked_files=True):
        if force_commit:
            raise Exception("force_commit: You have uncommitted (or untracked) work, commit them " +
                            "before running this script.")
    return repo.head.commit.hexsha

# Commits all changes (including untracked files, except those in .gitignore) in the repo specified
# by "autocommit_root" automatically, with the message "commit_message" and committer name "auto",
# if there are new changes. In the case "autocommit_root" is not a git repo, initializes one if
# force_create_repo is enabled, otherwise raises an exception. If new changes exist, returns the hash
# of the new commit, otherwise returns the latest commit hash. Works only in the active branch, if 
# 'push' is enabled, pushes to remote 'origin' on that branch.
def autocommit(autocommit_root=LEGGED_GYM_ROOT_DIR,  # Path to the root of the repo to autocommit
               force_create_repo=False,              # If enabled and the root is not a repo, creates it.
               commit_message="Autocommit",          # Message for each automatic commit.
               push=False):                          # If enabled, pushes commit to remote origin.
    # Find the absolute path for the repository for autocommit.
    autocommit_root = from_repo_root(autocommit_root)
    # Get the repo if it exists, else if force_create_repo is enabled, initialize the repo.
    # Otherwise, raise the error given by git.
    try:
        repo = Repo(autocommit_root)
    except InvalidGitRepositoryError:
        if force_create_repo:
            repo = Repo.init(autocommit_root)
        else:
            raise
    # Check if there are changes to this repo, if so make a new commit
    if repo.is_dirty(untracked_files=True):
        # Get logger to log the autocommit:
        log = logging.getLogger(__name__)
        # Add all changes to the index (including untracked files, excluding those in .gitignore)
        repo.git.add("-A")
        # Commit the added changes, naming the author "autocommit" (for easier filtering)
        commit = repo.index.commit(commit_message, committer=Actor("auto", "auto@commit.com"))
        # Print the newly committed files, up to 5 files, useful for seeing changes.
        # Save the changed files to be committed, in order to display them
        committed_files = list(commit.stats.files.keys())
        autocommit_log_str = f"New autocommit to {autocommit_root} with hash {commit.hexsha}: "
        if len(committed_files) > 5:
            autocommit_log_str += ", ".join(committed_files[:5]) + " and others..."
        else:
            autocommit_log_str += ", ".join(committed_files)
        log.info(autocommit_log_str)
        if push:  # Push if enabled
            repo.git.push()
            log.info("New commit pushed!")
    else:
        # Otherwise, output the latest commit hash
        print(f"Code not changed, no need to autocommit to {autocommit_root}")
    # Return the latest commit hash
    return repo.head.commit.hexsha

# Keeping a copy of the codebase in a specified path (by logs_root and autocommit_dir) in the 
# specified log folder, updates that copy and 'autocommits' new changes to the log folder.
# Used for code saving without disturbing the working repo. Pushes commits if enabled.
def codesave_in_logs(logs_root="../experiment_logs",   # The path to the specified log folder (absolute or relative from ground_control)           
                     codesave_dir="codesave",          # The directory path relative from 'logs_root' in which the codebase will be saved
                     commit_message="Autocommit",      # Message for each automatic commit
                     push=False):                      # If enabled, pushes new commits
    logs_root = from_repo_root(logs_root)  # Get absolute path of the experiment log folder
    codesave_full_path = os.path.join(logs_root, codesave_dir)  # Get absolute path to the code-saving directory
    os.makedirs(codesave_full_path, exist_ok=True)  # If any folder doesn't exist in the path to save, create it.
    
    # Backup this codebase to the code-saving directory.
    commands = ["rsync",  # used for incremental file transfer (only copying changed files)
                "-rc",  # recurse into folders, only copy when content has changed (not metadata)
                "--delete",  # delete files in destination if they're deleted in source 
                "--include='**.gitignore'",  # include .gitignore just in case we copy ignored files
                "--exclude='/.git'",  # exclude .git directory, not to clash with experiment repo
                "--filter=':- .gitignore'",  # ignore everything that git ignores
                # appending "/" to the source path for only copying content inside ground_control,
                # instead of the folder itself.
                LEGGED_GYM_ROOT_DIR + "/", codesave_full_path]
    result = subprocess.run(" ".join(commands), shell=True)
    # If there is an error in backup, raise that error.
    if result.stderr:
        raise subprocess.CalledProcessError(
                returncode = result.returncode,
                cmd = result.args,
                stderr = result.stderr
                )
    # Autocommit changes in the codebase backup to the experiment log repo, return the latest commit hash.
    return autocommit(logs_root, force_create_repo=True, commit_message=commit_message, push=push)
