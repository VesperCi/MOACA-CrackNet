 @Echo Off&SetLocal ENABLEDELAYEDEXPANSION
FOR %%a in (*) do (
echo 正在处理 %%a
set "name=%%a"
set "name=!name: =!"
set "name=!name:(=!"
set "name=!name:)=!"
ren "%%a" "!name!"
)
exit