Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll", SetLastError=true)]
    public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
}
"@

$windows = Get-Process | Where-Object { $_.MainWindowTitle -like '*Minecraft*' -and $_.MainWindowTitle -like '*1.21.3*' }
$cols = 3
$totalWindows = $windows.Count
$rows = [Math]::Ceiling($totalWindows / $cols)
$width = 360
$height = 360
$padding = -10
$index = 0

foreach ($proc in $windows) {
    $hwnd = $proc.MainWindowHandle
    if ($hwnd -ne 0) {
        $row = [Math]::Floor($index / $cols)
        $col = $index % $cols
        $X = ($width + $padding) * $col
        $Y = ($height + $padding) * $row
        [Win32]::MoveWindow($hwnd, $X, $Y, $width, $height, $true) | Out-Null
        Write-Host ("Positioned window $($index + 1): '$($proc.MainWindowTitle)' to ${width}x${height}px at (${X},${Y}).")
        $index++
    }
}

if ($index -lt $totalWindows) {
    Write-Host ("Warning: Only positioned $index of $totalWindows Minecraft windows.")
}