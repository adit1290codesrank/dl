@echo off
echo ===================================================
echo Automating Git Fix...
echo ===================================================

echo.
echo 1. Reverting the Amazon Summer Term remote branch back to before your push...
git push -f https://github.com/adit1290codesrank/deeplearning.git 7245d091fb2d3716a481d19a33500994aa0f351b:main

echo.
echo 2. Fixing your local 'origin' to point to your actual repository (dl.git)...
git remote set-url origin https://github.com/adit1290codesrank/dl.git

echo.
echo 3. Deleting untracked examples to fix your previous pull conflict...
del /Q examples\eval_alexa.cpp
del /Q examples\eval_cifar.cpp
del /Q examples\train_alexa.cpp
del /Q examples\train_cifar.cpp
del /Q examples\train_emnist.cpp

echo.
echo 4. Pulling the latest from your actual repository...
git pull origin main

echo.
echo 5. Pushing your current code to your actual repository...
git push origin main

echo.
echo ===================================================
echo Done! Everything is fixed.
echo ===================================================
pause
