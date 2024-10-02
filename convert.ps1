# Define the root directory where your videos are stored
$rootDirectory = "E:\dev\ISL\isl_detector\isl data\Greetings"

# Define the output directory for converted files
$outputRootDirectory = "E:\dev\ISL\isl_detector\isl data\Converted"

# Create the output directory if it doesn't exist
if (-not (Test-Path -Path $outputRootDirectory)) {
    New-Item -ItemType Directory -Path $outputRootDirectory
}

# Get all .mov files in the directory and its subdirectories
$movFiles = Get-ChildItem -Path $rootDirectory -Recurse -Filter *.mov

# Loop through each .mov file
foreach ($movFile in $movFiles) {
    # Define the relative path for the output file
    $relativePath = $movFile.FullName.Substring($rootDirectory.Length)
    $outputFile = Join-Path -Path $outputRootDirectory -ChildPath ([System.IO.Path]::ChangeExtension($relativePath, ".mp4"))

    # Create the output directory for the specific video if it doesn't exist
    $outputDirectory = [System.IO.Path]::GetDirectoryName($outputFile)
    if (-not (Test-Path -Path $outputDirectory)) {
        New-Item -ItemType Directory -Path $outputDirectory -Force
    }

    # Display the conversion message
    Write-Host "Converting '$($movFile.FullName)' to '$outputFile'..."

    # Use FFmpeg to convert the .mov file to .mp4
    ffmpeg -i "$($movFile.FullName)" -vf scale=1920:1080 "$outputFile"

    # Display completion message
    Write-Host "Conversion complete: '$outputFile'"
}

Write-Host "All .mov files have been converted to .mp4!"