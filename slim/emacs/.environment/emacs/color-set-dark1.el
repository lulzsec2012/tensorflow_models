;; =================================================================
;; Color setting.
;; Color is also inferenced by .Xdefault and emacs command option
;; =================================================================
(setq color-set-load t)

;; Set global font lock mode
(global-font-lock-mode '1)
;(global-font-lock-mode t)

;; Set foreground and background. black or dimgrey may be choosed
(set-foreground-color "white")
(set-background-color "Gray22")
;(set-foreground-color "black")
;(set-background-color "white")
;(set-background-color "CornflowerBlue")
;(set-background-color "DarkSlateBlue")
;(set-background-color "MidnightBlue")
;(set-background-color "Black")


;; Set the mouse and cursor color
(set-mouse-color "yellow")
(set-cursor-color "red")

;; Set manual face color
(set-face-foreground 'info-xref "cyan")
(set-face-foreground 'info-node "red")

(setq Man-overstrike-face 'info-node)
(setq Man-underline-face 'info-xref)

;; Set some font face
;(set-face-foreground 'font-lock-comment-face "red")
;(set-face-foreground 'font-lock-comment-face "Coral")
;(set-face-foreground 'font-lock-comment-face "darkgoldenrod")
;(set-face-foreground 'font-lock-comment-face "OrangeRed")
;(set-face-foreground 'font-lock-comment-face "DarkOrange")
(set-face-foreground 'font-lock-comment-face "LightSalmon")

(set-face-foreground 'font-lock-string-face "RosyBrown1")
;(set-face-foreground 'font-lock-string-face "plum1")
;(set-face-foreground 'font-lock-string-face "DeepSkyBlue")
;(set-face-foreground 'font-lock-string-face "OrangeRed4")
;(set-face-foreground 'font-lock-string-face "CadetBlue1")
;(set-face-foreground 'font-lock-string-face "CadetBlue3")

(set-face-foreground 'font-lock-variable-name-face "LightGoldenrod")
;(set-face-foreground 'font-lock-variable-name-face "darkblue")
;(set-face-foreground 'font-lock-variable-name-face "yellow")

(set-face-foreground 'font-lock-function-name-face "LightSkyBlue")
;(set-face-foreground 'font-lock-function-name-face "CadetBlue1")

(set-face-foreground 'font-lock-constant-face "Aquamarine")

(set-face-foreground 'font-lock-builtin-face "LightSteelBlue")
;(set-face-foreground 'font-lock-builtin-face "MediumOrchid4")
;(set-face-foreground 'font-lock-builtin-face "maroon3")

(eval-after-load "sh-script" 
  '(progn
     (set-face-foreground 'sh-heredoc-face "yellow3")))

(set-face-foreground 'font-lock-keyword-face "Cyan1")
;(set-face-foreground 'font-lock-keyword-face "darkviolet")
;(set-face-foreground 'font-lock-keyword-face "purple")

(set-face-foreground 'font-lock-type-face "PaleGreen")
;(set-face-foreground 'font-lock-type-face "ForestGreen")
;(set-face-foreground 'font-lock-type-face "darkgreen")


;===============window width and height===================
(set-frame-width (selected-frame) 75)
(set-frame-height (selected-frame)50)
