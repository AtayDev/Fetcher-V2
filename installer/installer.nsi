;===========================================
; Nordlys Fetcher Desktop - Installer Script
;===========================================

!define APP_NAME "Nordlys Fetcher"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "Your Company"
!define APP_URL "https://yourwebsite.com"
!define APP_EXE "NordlysFetcher.exe"
!define INSTALL_DIR "$PROGRAMFILES64\${APP_NAME}"

; Include Modern UI
!include "MUI2.nsh"

;===========================================
; General Settings
;===========================================
Name "${APP_NAME}"
OutFile "..\NordlysFetcher_Setup_v${APP_VERSION}.exe"
InstallDir "${INSTALL_DIR}"
InstallDirRegKey HKLM "Software\${APP_NAME}" "Install_Dir"
RequestExecutionLevel admin
ShowInstDetails show
ShowUnInstDetails show

;===========================================
; Interface Settings
;===========================================
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install-blue.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall-blue.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Wizard\orange.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Header\orange.bmp"

;===========================================
; Pages
;===========================================
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "README.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES

; Custom finish page with checkbox to run app
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Launch ${APP_NAME}"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README.txt"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Show setup instructions"
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;===========================================
; Languages
;===========================================
!insertmacro MUI_LANGUAGE "English"

;===========================================
; Version Information
;===========================================
VIProductVersion "${APP_VERSION}.0"
VIAddVersionKey "ProductName" "${APP_NAME}"
VIAddVersionKey "CompanyName" "${APP_PUBLISHER}"
VIAddVersionKey "FileVersion" "${APP_VERSION}"
VIAddVersionKey "ProductVersion" "${APP_VERSION}"
VIAddVersionKey "FileDescription" "${APP_NAME} Installer"
VIAddVersionKey "LegalCopyright" "© 2025 ${APP_PUBLISHER}"

;===========================================
; Installer Section
;===========================================
Section "Install"
    SetOutPath "$INSTDIR"
    
    ; Show detailed progress
    DetailPrint "Installing ${APP_NAME}..."
    
    ; Copy all files from dist folder
    DetailPrint "Copying application files..."
    File /r "..\dist\NordlysFetcher\*.*"
    
    ; Copy documentation
    DetailPrint "Installing documentation..."
    File "README.txt"
    
    ; Copy .env.example (if exists)
    IfFileExists "..\\.env.example" 0 +2
        File "..\\.env.example"
    
    ; Create .env from example if it doesn't exist
    DetailPrint "Setting up configuration..."
    IfFileExists "$INSTDIR\.env" skip_env_creation
        IfFileExists "$INSTDIR\.env.example" 0 skip_env_creation
            CopyFiles "$INSTDIR\.env.example" "$INSTDIR\.env"
    skip_env_creation:
    
    ; Create storage directories
    DetailPrint "Creating data directories..."
    CreateDirectory "$INSTDIR\storage"
    CreateDirectory "$INSTDIR\storage\chroma"
    CreateDirectory "$INSTDIR\storage\uploads"
    
    ; Create Start Menu shortcuts
    DetailPrint "Creating shortcuts..."
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\${APP_EXE}" 0
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\README.lnk" "$INSTDIR\README.txt"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
    
    ; Create Desktop shortcut
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\${APP_EXE}" 0
    
    ; Write registry keys
    DetailPrint "Registering application..."
    WriteRegStr HKLM "Software\${APP_NAME}" "Install_Dir" "$INSTDIR"
    WriteRegStr HKLM "Software\${APP_NAME}" "Version" "${APP_VERSION}"
    
    ; Write uninstall information
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "Publisher" "${APP_PUBLISHER}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "URLInfoAbout" "${APP_URL}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoRepair" 1
    
    ; Create uninstaller
    DetailPrint "Creating uninstaller..."
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
    DetailPrint "Installation complete!"
SectionEnd

;===========================================
; Uninstaller Section
;===========================================
Section "Uninstall"
    ; Remove files
    DetailPrint "Removing application files..."
    RMDir /r "$INSTDIR\*.*"
    
    ; Remove shortcuts
    DetailPrint "Removing shortcuts..."
    Delete "$DESKTOP\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\*.*"
    RMDir "$SMPROGRAMS\${APP_NAME}"
    
    ; Ask about user data
    MessageBox MB_YESNO|MB_ICONQUESTION "Do you want to delete your documents and database?$\n$\n\
        Your uploaded files: $INSTDIR\storage\uploads\$\n\
        Your database: $INSTDIR\storage\chroma\$\n$\n\
        Choose 'No' to keep your data for future reinstall." IDYES delete_data IDNO skip_data
    
    delete_data:
        DetailPrint "Removing user data..."
        RMDir /r "$INSTDIR\storage"
        Delete "$INSTDIR\.env"
    skip_data:
    
    ; Remove installation directory
    RMDir "$INSTDIR"
    
    ; Remove registry keys
    DetailPrint "Cleaning registry..."
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
    DeleteRegKey HKLM "Software\${APP_NAME}"
    
    DetailPrint "Uninstall complete!"
SectionEnd