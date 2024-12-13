@echo off
for /L %%i in (1,1,9) do (
    start cmd /k "cd /d E:\mcmodding\fabric-example-mod && gradlew runClient%%i"
)