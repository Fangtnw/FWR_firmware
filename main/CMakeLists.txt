
idf_component_register(
    SRCS  "ofd.cpp"  "yolo.cpp" "fwr_control/fwr_control.cpp"  "fwr_control/sbus_rx.cpp"
    # "main.cpp"
    # "main_rx_test.cpp"
    # "main_fwr_video.cpp"
    # "main_fwr_test.cpp"
    "main_fwr_ofd.cpp"
   
    #  "espdet_detect.cpp" "espdet_detect.hpp" "espdet.cpp"
    INCLUDE_DIRS "." "fwr_control"
    REQUIRES
        esp_video
        esp_cam_sensor
        esp_sccb_intf
        esp_driver_cam
        esp_driver_uart
        esp_lcd
        esp_lcd_ek79007
        driver
        esp_timer
        unity
        example_video_common 
        esp-dl
        lcd
        camera
        fatfs
        vfs
        sdmmc
    EMBED_FILES
        # "model/best_esp32p4_320.espdl"
        #  "model/basketball_640.espdl"
        #  "model/basketball_64.espdl"
        # "model/kyubest1.espdl"   #320x320
        # "model/ESPdet.espdl"   #160x160
        "model/bas.espdl"   #160x160
        # "model/best_esp32p4_320.info"
        # "model/best_esp32p4_320.json"
)


target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -ffast-math)
