;; emacs initial for CPU group
;; no tab
(global-set-key "\M-s" 'save-buffer-no-tab)
(defun save-buffer-no-tab ()
    "Replace tab with space and then save-buffer"
    (interactive)
    (untabify (point-min) (point-max))
    (save-buffer))
;; insert time
(global-set-key "\M-t" 'insert-current-time )
(defun insert-current-time()
    (interactive)
    (template-insert-time "%Y-%m-%d:%T" "0000-00-00:00:00:00")
)

;; config verilog-mode
(setq
      verilog-indent-level             4
      verilog-indent-level-module      4
      verilog-indent-level-declaration 4
      verilog-indent-level-behavioral  4
      verilog-auto-newline             nil
      verilog-auto-indent-on-newline   nil
      verilog-tab-always-indent        nil
      verilog-auto-inst-dot-name       t
)

