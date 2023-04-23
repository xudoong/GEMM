for mnk in 240, 384, 480, 768, 960, 1200, 1440, 1920, 2400, 3840
do
    taskset 1 ./main.x ${mnk}
    echo ""
done