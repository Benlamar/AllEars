$voice = New-Object -ComObject SAPI.SpVoice
$fileStream = New-Object -ComObject SAPI.SpFileStream
$fileStream.Open("data\speech_sample.wav", 3)
$voice.AudioOutputStream = $fileStream
$voice.Speak("Welcome to the local audio intelligence system. The quick brown fox jumps over the lazy dog.")
$fileStream.Close()
Write-Host "Created data\speech_sample.wav successfully"

