;; =================================================================
;; Color setting.
;; Color is also inferenced by .Xdefault and emacs command option
;; =================================================================
(setq color-set-load t)

;; Set global font lock mode
(global-font-lock-mode '1)
;(global-font-lock-mode t)

;; Set foreground and background. black or dimgrey may be choosed
(set-foreground-color "black")
(set-background-color "lightyellow")

;; Set the mouse and cursor color
(set-mouse-color "black")
(set-cursor-color "red")

;; Set manual face color
(set-face-foreground 'info-xref "cyan")
(set-face-foreground 'info-node "red")

(setq Man-overstrike-face 'info-node)
(setq Man-underline-face 'info-xref)

;; Set some font face
(set-face-foreground 'font-lock-comment-face "royalblue")
(set-face-foreground 'font-lock-string-face "blue")
(set-face-foreground 'font-lock-variable-name-face "darkorchid")
(set-face-foreground 'font-lock-function-name-face "darkblue")
(set-face-foreground 'font-lock-constant-face "cyan4")
(set-face-foreground 'font-lock-builtin-face "darkgreen")

;(load "sh-script")
;(set-face-foreground 'sh-heredoc-face "royalblue")
(eval-after-load "sh-script" 
  '(progn
     (set-face-foreground 'sh-heredoc-face "blue")))

(set-face-foreground 'font-lock-keyword-face "red4")
(set-face-foreground 'font-lock-type-face "magenta")

