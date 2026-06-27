SKILL_DIR := skills/pytrio-skill
DIST_DIR := dist
ARCHIVE_NAME := pytrio-skill

.PHONY: package pack clean

package: clean
	mkdir -p $(DIST_DIR)
	cd $(SKILL_DIR) && tar --exclude='__pycache__' --exclude='*.pyc' -czf ../../$(DIST_DIR)/$(ARCHIVE_NAME).tar.gz .
	cd $(SKILL_DIR) && zip -qr ../../$(DIST_DIR)/$(ARCHIVE_NAME).zip . -x '*/__pycache__/*' '*.pyc'
	@echo "已生成 $(DIST_DIR)/$(ARCHIVE_NAME).tar.gz"
	@echo "已生成 $(DIST_DIR)/$(ARCHIVE_NAME).zip"

pack: package

clean:
	rm -rf $(DIST_DIR)
