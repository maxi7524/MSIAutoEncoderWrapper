#TODO - opisać dokładnei jak przechodzić przez tutoriale 





 
## fragmetny krtóre powinny ogólnie opisaywac co się dzieje w notebookach 

### 1. Initialize the tutorial environment

#### 1.1. Resolve the repository root

The notebook expects to run inside the project repository. 

The cell below finds the nearest parent directory containing `pyproject.toml` and makes it the working directory. This keeps relative cache and configuration paths independent of the directory from which Jupyter was started.