@echo off
SET REMOTE_USER=pmatlia
SET REMOTE_HOST=cantor.math.purdue.edu
SET REMOTE_BASE=/home/pmatlia/feynman_kac
SET LOCAL_DIR=D:\Research\Feynman-Kac-2025
SET SSH_KEY=C:\Users\purav\.ssh\cantor

REM --- Step 1: Move excluded files out of data/AAPL ---
ssh -i "%SSH_KEY%" %REMOTE_USER%@%REMOTE_HOST% "mv %REMOTE_BASE%/data/AAPL/aapl_stock.csv %REMOTE_BASE%/; mv %REMOTE_BASE%/data/AAPL/aapl_options.csv %REMOTE_BASE%/"

REM --- Step 2: Copy data/ and models/ recursively ---
scp -i "%SSH_KEY%" -r %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_BASE%/data %LOCAL_DIR%
scp -i "%SSH_KEY%" -r %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_BASE%/models %LOCAL_DIR%

REM --- Step 3: Move the CSV files back to data/AAPL ---
ssh -i "%SSH_KEY%" %REMOTE_USER%@%REMOTE_HOST% "mv %REMOTE_BASE%/aapl_stock.csv %REMOTE_BASE%/data/AAPL/; mv %REMOTE_BASE%/aapl_options.csv %REMOTE_BASE%/data/AAPL/"

echo Sync complete!
pause
