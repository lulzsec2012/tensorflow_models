;; PC Linux common emacs site start
;;

;; Enable CUA mode
;; This maps edit keys to standard Windows keystokes. It requires the
;; library cua-mode.el from Kim Storm at the following URL:
;(load "cua-mode" t)
;(CUA-mode t)
(if (load "cua-mode" t) (CUA-mode t))


;; Enable verilog mode font lock
;; Load verilog mode only when needed
(autoload 'verilog-mode "verilog-mode" "Verilog mode" t )
;; Any files that end in .v should be in verilog mode
(setq auto-mode-alist (cons  '("\\.v\\'" . verilog-mode) auto-mode-alist))
;; Any files in verilog mode should have their keywords colorized
;(add-hook 'verilog-mode-hook '(lambda () (font-lock-mode 1)))

;; Enable dcsh mode font lock
(autoload 'dcsh-mode "dcsh-mode" "Dcsh Mode" t)
(setq auto-mode-alist (cons '("\\.scr\\'" . dcsh-mode) auto-mode-alist))

;; Enable specman mode font lock
(autoload 'specman-mode "specman-mode" "Specman code editing mode" t) 
(setq auto-mode-alist 
	  (append (list 
		   (cons "\\.e\\'" 'specman-mode)
		   (cons "\\.e3\\'" 'specman-mode)
		   (cons "\\.load\\'" 'specman-mode)
		   (cons "\\.ecom\\'" 'specman-mode)
		   (cons "\\.etst\\'" 'specman-mode))
		  auto-mode-alist))                     
