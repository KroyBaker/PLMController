from pypylon import pylon
import cv2

camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)

camera.Open()

print("Camera:", camera.GetDeviceInfo().GetModelName())

camera.PixelFormat.SetValue("Mono8")
camera.ExposureAuto.SetValue("Off")
camera.GainAuto.SetValue("Off")

camera.CenterX.SetValue(False)
camera.CenterY.SetValue(False)

camera.Width.SetValue(160)
camera.Height.SetValue(60)


camera.OffsetX.SetValue(128)
camera.OffsetY.SetValue(240)

camera.ExposureTime.SetValue(81.0)  # microseconds, adjust as needed
camera.Gain.SetValue(0.0)
camera.AcquisitionFrameRateEnable.SetValue(False)


fps_limit = camera.ResultingFrameRate.GetValue()
print(f"New Resulting Frame Rate: {fps_limit:.2f} FPS")

window_name = "Basler acA640-750um"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

try:
    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        try:
            if grab.GrabSucceeded():
                img = grab.Array
                cv2.imshow(window_name, img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            grab.Release()
finally:
    if camera.IsGrabbing():
        camera.StopGrabbing()
    if camera.IsOpen():
        camera.Close()
    cv2.destroyAllWindows()
